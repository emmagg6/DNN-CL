%% emmagg6 - added since did not recognize in the Matlab 2024a for Apple Silicon setup

function hash = GetMD5(data, varargin)
    % Convert data to a byte stream
    byteStream = getByteStreamFromArray(data);
    
    % Compute the MD5 hash
    md = java.security.MessageDigest.getInstance('MD5');
    md.update(byteStream);
    hash = reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []);
    
    % Process optional arguments (e.g., 'array', 'hex')
    if nargin > 1
        if strcmp(varargin{1}, 'array')
            hash = uint8(md.digest());
        elseif strcmp(varargin{1}, 'hex')
            hash = reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []);
        end
    end
end