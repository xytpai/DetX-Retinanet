from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools


__version__ = '0.0.1'


class get_pybind_include(object):
    # 找到pybind11路径
    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'dbext', # 包名
        sorted(['dbext.cpp']), # 源文件名
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++'
    ),
]


def has_flag(compiler, flagname):
    # 用来测试环境是否支持编译
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11'] # flags选项
    # flags = ['-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    c_opts = { # 编辑编译选项
        'msvc': ['/EHsc'], # Windows
        'unix': [], # Linux
    }
    l_opts = { # 编辑链接选项
        'msvc': [],
        'unix': [],
    }
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', 
                '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='dbext', # 包名
    version=__version__,
    author='xytpai',
    author_email='xytpai@foxmail.com',
    url='https://github.com/xytpai',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)