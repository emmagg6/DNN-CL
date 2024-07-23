% Adapted from the Brain Engineering Lab publication "A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture" by Rodriguez A, Bowen EFW, Granger R (2022)


function [pixels, label_idx, uniq_classes, pixel_metadata, label_metadata, other_metadata] = LoadFashionMNIST(spec, is_trn, n_per_class)
    arguments
        spec(1,1) string, is_trn(1,1) logical, n_per_class(1,1)
    end
    pixel_metadata = struct();
    label_metadata = struct();
    other_metadata = struct();

    if spec == "fashionmnistpy"
        % clear classes
        python_code = py.importlib.import_module("python_code");
        py.importlib.reload(python_code);
        if is_trn
            if isnan(n_per_class)
                x = struct(python_code.dataset(is_trn, int64(-1)));
            else
                assert(n_per_class <= 6000, "n_per_class must be <= 6000 (or NaN = all)");
                x = struct(python_code.dataset(is_trn, int64(n_per_class)));
            end
        else
            x = struct(python_code.dataset(is_trn, int64(-1)));
        end
        pixels = logical(x.data); % n_trn x 784 (comes in as uint8)
        labelIdx = double(x.label_idx) + 1; % 1 x n_trn (comes in as uint8) indexes into uniq_classes
        pixels = pixels';
        img = reshape(pixels, 28, 28, []);
        img = permute(img, [2,1,3]); % the data from python are transposed
        img = reshape(img, 28*28, []);
        dataset = struct();
        dataset.sense = {img};
        dataset.class_idx = labelIdx;
        dataset.uniq_class = {'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'};
    elseif spec == "fashionmnistmat"
        if is_trn
            assert(isnan(n_per_class) || n_per_class <= 6000, "n_per_class must be <= 6000 (or NaN = all)");
            load(fullfile(Config.DATASET_DIR, "img_captchas", "fashionmnist.trn-eqn-img-vec-noise.mat"), "dataset");
        else
            load(fullfile(Config.DATASET_DIR, "img_captchas", "fashionmnist.tst-eqn-img-vec-noise.mat"), "dataset");
        end
    else
        error("unexpected spec");
    end

    pixels       = dataset.sense{1};
    label_idx    = dataset.class_idx;
    uniq_classes = dataset.uniq_class;

    [row,col] = PixelRowCol([28,28,1]);
    pixel_metadata.name = cell(size(pixels, 1), 1);
    for j = 1 : size(pixels, 1)
        pixel_metadata.name{j} = ['px_r',num2str(row(j)),'_c',num2str(col(j))];
    end

    % equalize n
    if is_trn && (spec ~= "fashionmnistpy")
        idx = EqualizeN(label_idx, n_per_class); % permanently reduce the number of images, while equalizing N
    else
        idx = EqualizeN(label_idx); % just equalize N
    end
    pixels = pixels(:,idx);
    label_idx = label_idx(idx);

    pixel_metadata.chanidx = ones(size(pixels, 1), 1);
end