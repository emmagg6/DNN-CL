%%% Random seed function

function SetRNG(cfg)
    % Check if cfg is a structure with a field 'seed'
    % if isstruct(cfg) && isfield(cfg, 'seed')
    %     rng(cfg.seed); % Initialize RNG with the provided seed
    %     fprintf('RNG initialized to seed = %d\n', cfg.seed);
    % else
    %     disp('cfg contents:');
    %     disp(cfg); % Display the contents of cfg
    %     error('cfg must be a structure with a field ''seed''');
    % end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% E
    disp(cfg)
    rng(1000)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end