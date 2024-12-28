import os

def massbank_to_mgf(input_folder, output_mgf_file):
    """
    MassBank形式の複数のテキストファイルを1つのMGFファイルに変換する。

    Parameters:
        input_folder (str): MassBank形式のテキストファイルが格納されたフォルダのパス。
        output_mgf_file (str): 出力するMGFファイルのパス。
    """
    def parse_massbank_record(record_text):
        lines = record_text.strip().split("\n")
        precursor_mz = None
        mz_values = []
        intensity_values = []

        in_peak_section = False
        for line in lines:
            if line.startswith("MS$FOCUSED_ION: PRECURSOR_M/Z"):
                precursor_mz = float(line.split()[-1])
            elif line.startswith("PK$PEAK:"):
                in_peak_section = True
            elif in_peak_section:
                if line.startswith("//"):
                    break
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        mz_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        pass
        return precursor_mz, mz_values, intensity_values

    # MGFファイルの書き込みを開始
    with open(output_mgf_file, "w") as mgf_file:
        for idx, filename in enumerate(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)

                with open(file_path, "r") as f:
                    record_text = f.read()

                precursor_mz, mz_values, intensity_values = parse_massbank_record(record_text)

                if precursor_mz is not None and mz_values:
                    # MGFフォーマットのスペクトルを書き込む
                    mgf_file.write("BEGIN IONS\n")
                    mgf_file.write(f"TITLE=Spectrum_{idx + 1}\n")
                    mgf_file.write(f"PEPMASS={precursor_mz}\n")

                    for mz, intensity in zip(mz_values, intensity_values):
                        mgf_file.write(f"{mz} {intensity}\n")

                    mgf_file.write("END IONS\n\n")

    print(f"MGF file saved to {output_mgf_file}")

# 実行例
input_folder = "C:/Users/hyama/Documents/LargeScaleMSPy/MassBank-Human-main/HSA001"
output_mgf_file = "C:/Users/hyama/data/combined_spectra.mgf"
massbank_to_mgf(input_folder, output_mgf_file)
