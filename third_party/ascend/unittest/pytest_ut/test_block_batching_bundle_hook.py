# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise the packaged public compiler hook, including callbacks deferred by JIT."""
import ast
from pathlib import Path
import shutil
import subprocess
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='module')
def packaged_entries(tmp_path_factory):
    root = Path(__file__).resolve().parents[4]
    staging = tmp_path_factory.mktemp('bundle-hook')
    files = ('python/triton/compiler/compiler.py', 'python/triton/runtime/jit.py')
    for name in files:
        out = staging / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, out)
    # CI can run from the already-patched worktree. Otherwise apply the same
    # distribution patch as the package build, restricted to the two inputs.
    if 'def _compile(' not in (staging / files[0]).read_text():
        patch = root / 'third_party/ascend/patch/triton-ascend-3.6.0.patch'
        subprocess.run(['git', 'apply', *('--include=' + f for f in files), str(patch)], cwd=staging, check=True)
    compiler, jit = [ast.parse((staging / name).read_text()) for name in files]
    entry = next(n for n in compiler.body if isinstance(n, ast.FunctionDef) and n.name == 'compile')
    jit_cls = next(n for n in jit.body if isinstance(n, ast.ClassDef) and n.name == 'JITFunction')
    do_compile = next(n for n in jit_cls.body if isinstance(n, ast.FunctionDef) and n.name == '_do_compile')
    return (compile(ast.Module(body=[entry], type_ignores=[]), str(staging / files[0]),
                    'exec'), compile(ast.Module(body=[do_compile], type_ignores=[]), str(staging / files[1]), 'exec'))


@pytest.fixture
def compiler(packaged_entries):
    target = object()
    backend = SimpleNamespace()
    make_backend = Mock(return_value=backend)
    single = Mock(side_effect=lambda src, target, options, env: (src, target, options, env))
    getenv = Mock(return_value={'KEY': 'current'})
    scope = dict(make_backend=make_backend, GPUTarget=object, _compile=single, get_cache_invalidating_env_vars=getenv,
                 driver=SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target)))
    exec(packaged_entries[0], scope)
    return SimpleNamespace(compile=scope['compile'], backend=backend, make_backend=make_backend, single=single,
                           getenv=getenv, target=target)


@pytest.mark.parametrize('explicit', [None, {}, {'KEY': 'submitted', 'CUSTOM': 'caller-only'}])
def test_opt_in_hook_freezes_one_environment_for_candidate_and_baseline(compiler, explicit):
    caller_env = None if explicit is None else dict(explicit)
    expected = dict(compiler.getenv.return_value if caller_env is None else caller_env)
    frozen = object()
    baseline_options = {'block_batching_factor': 1}

    def with_env(src, options, compile_baseline, *, env_vars):
        # A hook may take time to freeze the source, or invoke a callback after
        # it returns. Neither the process nor the user's map can alter this request.
        (compiler.getenv.return_value if caller_env is None else caller_env)['KEY'] = 'later'
        return env_vars, compile_baseline

    with_env.accepts_env_vars = True
    compiler.backend.compile_bundle = with_env
    env, baseline = compiler.compile('original', options={'block_batching_factor': 4}, _env_vars=caller_env)
    assert env == expected
    assert baseline(baseline_options, frozen) == (frozen, compiler.target, baseline_options, expected)
    assert baseline(baseline_options) == ('original', compiler.target, baseline_options, expected)
    assert compiler.getenv.call_count == int(caller_env is None)


@pytest.mark.parametrize('caller_env', [None, {}, {'OLD': 'value'}])
def test_existing_three_argument_hook_preserves_callback_semantics(compiler, caller_env):

    def legacy(src, options, compile_variant):
        return compile_variant

    compiler.backend.compile_bundle = legacy
    opt = {'block_batching_factor': 2}
    callback = compiler.compile('original', options=opt, _env_vars=caller_env)
    if caller_env is not None:
        caller_env['LATER'] = 'legacy caller'
    assert callback(opt, 'frozen') == ('frozen', compiler.target, opt, caller_env)
    assert callback(opt)[3] is caller_env
    compiler.getenv.assert_not_called()


@pytest.mark.parametrize('marker', [None, 'yes'])
def test_legacy_override_does_not_inherit_parent_method_opt_in(compiler, marker):

    class OptIn:

        def compile_bundle(self, src, options, baseline, *, env_vars):
            pytest.fail('the inherited method must not override the selected backend method')

        compile_bundle.accepts_env_vars = True

    class Legacy(OptIn):

        def compile_bundle(self, src, options, baseline):
            return baseline

    if marker is not None:
        Legacy.compile_bundle.accepts_env_vars = marker
    compiler.make_backend.return_value = Legacy()
    caller_env = {'EXPLICIT': 'legacy'}
    callback = compiler.compile('original', options={'block_batching_factor': 2}, _env_vars=caller_env)
    assert callback({}) == ('original', compiler.target, {}, caller_env)
    compiler.getenv.assert_not_called()


@pytest.mark.parametrize('options,has_hooks', [
    (None, True),
    ({}, True),
    ({'block_batching_factor': 1}, True),
    ({'block_batching_factor': 4}, False),
])
def test_single_kernel_bypasses_bundle_hooks_and_forwards_environment(compiler, options, has_hooks):
    if has_hooks:
        compiler.backend.compile_bundle = Mock(side_effect=AssertionError)
    caller_env = {}
    src = object()
    assert compiler.compile(src, options=options, _env_vars=caller_env) == (src, compiler.target, options, {})
    assert compiler.single.call_args.args[3] is caller_env
    compiler.getenv.assert_not_called()


def test_async_jit_submits_and_compiles_with_the_same_environment(packaged_entries, compiler):
    submission = Mock(return_value='pending')

    def compile_bundle(src, opts, base, *, env_vars):
        return base({'block_batching_factor': 1})

    compile_bundle.accepts_env_vars = True
    compiler.backend.compile_bundle = compile_bundle
    cache = {}
    kernel = SimpleNamespace(device_caches={0: (cache, {}, compiler.target, compiler.backend, None)},
                             _call_hook=lambda *args: False, ASTSource=lambda *args: 'ttir-source',
                             compile=compiler.compile)
    current = {'KEY': 'submitted'}
    options = SimpleNamespace(block_batching_factor=4)
    key = Mock(return_value='submission-key')
    active = ContextVar('test_async_compile_mode', default=SimpleNamespace(submit=submission))
    jit_scope = dict(knobs=SimpleNamespace(runtime=SimpleNamespace(jit_cache_hook=None, jit_post_compile_hook=None)),
                     _async_compile=SimpleNamespace(active_mode=active),
                     get_cache_invalidating_env_vars=lambda: dict(current), get_cache_key=key)
    exec(packaged_entries[1], jit_scope)
    assert jit_scope['_do_compile'](kernel, 'jit-key', {}, 0, {}, options, {}, False) == 'pending'
    key.assert_called_once_with('ttir-source', compiler.backend, options, {'KEY': 'submitted'})
    current['KEY'] = 'changed after submission'
    submission_key, deferred, finalize = submission.call_args.args
    assert submission_key == 'submission-key'
    compiled = deferred()
    assert compiled == ('ttir-source', compiler.target, {'block_batching_factor': 1}, {'KEY': 'submitted'})
    finalize(compiled)
    assert cache == {'jit-key': compiled}
    compiler.getenv.assert_not_called()
