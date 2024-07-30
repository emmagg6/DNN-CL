% emmagg6 - wrapper function or train_liblinear_multicore due to errors

% % CHECK
% if exist('train_liblinear_multicore', 'file')
%     disp('train_liblinear_multicore is accessible.');
% else
%     error('train_liblinear_multicore is not accessible. Ensure it is in the MATLAB path.');
% end

function model = train_liblinear_multicore(label, data, options)
    % A wrapper function for train_liblinear
    model = train_liblinear(label, sparse(data), options);
end