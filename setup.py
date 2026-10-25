from setuptools import find_packages, setup


setup(
    name="bilgeval",
    version="0.2.0",
    description="Standalone LLM evaluation against OpenAI-compatible endpoints.",
    packages=find_packages(include=["bilgeval", "bilgeval.*"]),
    include_package_data=True,
    package_data={
        "bilgeval": [
            "builtin_registry.json",
            "catalog/*.yaml",
            "instruction_following/NOTICE.md",
            "instruction_following/LICENSE-APACHE-2.0",
            "instruction_following/nltk_data/**/*",
        ]
    },
    install_requires=[
        "openai>=2.21,<4",
        "httpx>=0.25,<1",
        "numpy>=1.24,<3",
        "pydantic>=2.7,<3",
    ],
    extras_require={
        "td": ["tokenizers>=0.23,<0.24"],
        "all": [
            "boto3>=1.28",
            "pyarrow>=15,<24",
            "nltk>=3.9",
            "langdetect>=1.0.9",
            "immutabledict>=4",
            "emoji>=2",
            "syllapy>=0.8.0",
            "starlette>=0.40,<1",
            "uvicorn>=0.30,<1",
            "aiohttp>=3.13.3,<4",
            "httpx-aiohttp>=0.2,<1",
        ],
        "dev": ["pytest>=8,<9", "pyyaml>=6,<7"],
        "ifeval": [
            "nltk>=3.9",
            "langdetect>=1.0.9",
            "immutabledict>=4",
            "emoji>=2",
            "syllapy>=0.8.0",
        ],
        "parquet": ["pyarrow>=15,<24"],
        "s3": ["boto3>=1.28"],
        "proxy": ["starlette>=0.40,<1", "uvicorn>=0.30,<1", "aiohttp>=3.13.3,<4"],
        "async": ["aiohttp>=3.13.3,<4", "httpx-aiohttp>=0.2,<1"],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "bilgeval=bilgeval.main:main",
            "bilgeval-proxy=bilgeval.proxy:main",
        ]
    },
)
