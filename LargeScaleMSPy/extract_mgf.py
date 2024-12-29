import os

def massbank_to_mgf(input_folder, output_mgf_file):
    """
    MassBank形式の複数のテキストファイルを1つのMGFファイルに変換する。
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

    with open(output_mgf_file, "w") as mgf_file:
        for idx, filename in enumerate(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)

                with open(file_path, "r") as f:
                    record_text = f.read()

                precursor_mz, mz_values, intensity_values = parse_massbank_record(record_text)

                if precursor_mz is not None and mz_values:
                    mgf_file.write("BEGIN IONS\n")
                    mgf_file.write(f"TITLE=Spectrum_{idx + 1}\n")
                    mgf_file.write(f"PEPMASS={precursor_mz}\n")
                    for mz, intensity in zip(mz_values, intensity_values):
                        mgf_file.write(f"{mz} {intensity}\n")
                    mgf_file.write("END IONS\n\n")

    print(f"MGF file saved to {output_mgf_file}")

def parse_mgf_file(file_path):
    """
    MGFファイルを解析してスペクトルデータを取得する。
    """
    spectra = []
    with open(file_path, "r") as f:
        precursor_mz = None
        mz_values = []
        intensity_values = []
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                precursor_mz = None
                mz_values = []
                intensity_values = []
            elif line.startswith("PEPMASS"):
                precursor_mz = float(line.split("=")[1])
            elif line.startswith("END IONS"):
                if precursor_mz is not None and mz_values:
                    spectra.append((precursor_mz, np.array(mz_values), np.array(intensity_values)))
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        continue
    return spectra

def save_mgf_to_text(mgf_data, output_file):
    """
    MGFデータをテキスト形式で保存する。
    """
    with open(output_file, "w") as f:
        for idx, (precursor_mz, mz_values, intensity_values) in enumerate(mgf_data):
            f.write(f"BEGIN IONS\n")
            f.write(f"TITLE=Spectrum_{idx + 1}\n")
            f.write(f"PEPMASS={precursor_mz}\n")
            for mz, intensity in zip(mz_values, intensity_values):
                f.write(f"{mz} {intensity}\n")
            f.write(f"END IONS\n\n")
    print(f"MGF data saved to {output_file}")
