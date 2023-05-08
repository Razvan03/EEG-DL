%%
function save_data_in_chunks(file_prefix, data)
    chunk_size = 500; % You can adjust the chunk size based on the available memory
    num_chunks = ceil(size(data, 1) / chunk_size);
for i = 1:num_chunks
        start_row = (i - 1) * chunk_size + 1;
        end_row = min(i * chunk_size, size(data, 1));
        data_chunk = data(start_row:end_row, :);
        file_name = sprintf('%s_chunk_%d.xlsx', file_prefix, i);
        xlswrite(file_name, data_chunk);
    end
end