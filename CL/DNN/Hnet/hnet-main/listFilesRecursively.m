function listFilesRecursively(directory)
    % List all files and folders in the directory
    files = dir(directory);
    
    % Loop through each file/folder
    for i = 1:length(files)
        % Get the full path of the file/folder
        fullPath = fullfile(files(i).folder, files(i).name);
        
        % Check if it is a directory
        if files(i).isdir
            % Skip the current and parent directory entries
            if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
                continue;
            end
            
            % Display the directory
            fprintf('Directory: %s\n', fullPath);
            
            % Recursively list files in the subdirectory
            listFilesRecursively(fullPath);
        else
            % Display the file
            fprintf('File: %s\n', fullPath);
        end
    end
end