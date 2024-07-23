%%% Random seed function

function SetRNG(cfg)
    % Check if cfg is a structure with a field 'seed'
    if isstruct(cfg) && isfield(cfg, 'seed')
        rng(cfg.seed); % Initialize RNG with the provided seed
        fprintf('RNG initialized to seed = %d\n', cfg.seed);
    else
        error('cfg must be a structure with a field ''seed''');
    end
end