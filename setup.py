from setuptools import setup, find_packages

setup(
    name="LargeScaleMSPy",  # パッケージ名
    version="1.0.0",  # バージョン番号
    description="MS/MS spectrum visualization by UMAP",  # 簡単な説明
    author="Hiroyuki Yamamoto",  # 作者名
    author_email="h.yama2396@gmail.com",  # 作者のメールアドレス
    url="https://github.com/hiroyukiyamamoto/LargeScaleMSPy",  # GitHubリポジトリのURL
    packages=find_packages(),  # パッケージを自動検出
    include_package_data=True,  # 非Pythonファイルも含める
    install_requires=[
        "numpy",
        "scipy",
        "h5sparse",
        "pandas",
        "plotly",
        "dash",
        "umap-learn"
    ],
    entry_points={
        "console_scripts": [
            "umap-msms-viewer=LargeScaleMSPy.main:run_app"  # 実行コマンドのエントリーポイント
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # サポートするPythonバージョン
)
