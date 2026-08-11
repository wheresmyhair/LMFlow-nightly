import os

from setuptools import find_packages, setup

folder = os.path.dirname(__file__)
version_path = os.path.join(folder, "src", "lmflow", "version.py")

__version__ = None
with open(version_path) as f:
    exec(f.read(), globals())

req_path = os.path.join(folder, "requirements", "base.txt")
with open(req_path, encoding="utf-8") as fp:
    install_requires = [
        line.strip()
        for line in fp
        if line.strip() and not line.lstrip().startswith("#")
    ]

extra_require = {
    "multimodal": ["Pillow"],
    "vllm": ["vllm>=0.8.0"],
    # pybase64 is imported eagerly by sglang.utils but not declared as a hard
    # dep upstream; without it `import sglang` raises ModuleNotFoundError.
    "sglang": ["sglang", "pybase64"],
    "ray": ["ray>=2.22.0"],
    "gradio": ["gradio"],
    "flask": ["flask", "flask_cors"],
    "flash_attn": ["flash-attn>=2.0.2"],
    # rich is lazy-imported by trl's DPOTrainer; not declared in trl 0.11.x.
    "trl": ["trl>=0.11,<0.12", "rich"],
    "deepspeed": ["deepspeed>=0.14.4"],
    "develop": ["pytest"],
    "dev": ["ruff", "pytest", "pre-commit"],
}

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as fp:
        readme_contents = fp.read().strip()

setup(
    name="lmflow",
    version=__version__,
    description="LMFlow: Large Model Flow.",
    author="The LMFlow Team",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={},
    install_requires=install_requires,
    extras_require=extra_require,
    classifiers=[
        "Intended Audience :: Science/Research/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
)

# optionals
# lm-eval==0.3.0
