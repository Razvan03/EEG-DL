import pandas as pd
import glob

def read_excel_chunks(file_prefix, output_file):
    chunk_files = sorted(glob.glob(file_prefix + "_chunk_*.xlsx"))
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter', options={'use_zip64': True})

    start_row = 0
    for file in chunk_files:
        data_chunk = pd.read_excel(file)
        data_chunk.to_excel(writer, startrow=start_row, index=False)

        start_row += len(data_chunk) + 1

    writer.save()

read_excel_chunks('training_set', 'training_set_concatenated.xlsx')
read_excel_chunks('test_set', 'test_set_concatenated.xlsx')
read_excel_chunks('training_label', 'training_label_concatenated.xlsx')
read_excel_chunks('test_label', 'test_label_concatenated.xlsx')
read_excel_chunks('all_data', 'all_data_concatenated.xlsx')
read_excel_chunks('all_labels', 'all_labels_concatenated.xlsx')