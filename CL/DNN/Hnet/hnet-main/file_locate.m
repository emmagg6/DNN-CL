% Define the directory path
directoryPath = '/Users/emmagraham/Documents/MATLAB/hnet-main';

% Get list of all files and directories
fileList = dir(fullfile(directoryPath, '**', '*'));

% Loop through and display each file and directory
for i = 1:length(fileList)
    % Skip the current and parent directory entries
    if fileList(i).isdir && (strcmp(fileList(i).name, '.') || strcmp(fileList(i).name, '..'))
        continue;
    end
    
    % Get the full path of the file/directory
    fullPath = fullfile(fileList(i).folder, fileList(i).name);
    
    % Display the file/directory
    if fileList(i).isdir
        fprintf('Directory: %s\n', fullPath);
    else
        fprintf('File: %s\n', fullPath);
    end
end