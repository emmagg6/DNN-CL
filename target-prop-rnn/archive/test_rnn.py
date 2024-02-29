from rnn import *


def test_get_dataloader():
    dataloaders = utils.get_dataloaders(batch_size=20, seq_len=20, train_batch_num=1000,
                                  val_batch_num=500)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    train_batch_num = 0
    val_batch_num = 0

    train_X, train_Y = next(train_loader)

    assert(train_X.shape == torch.Size([20, 20, 4]))
    assert(train_X.dtype == torch.float)

    assert(train_Y.shape == torch.Size([20, 20]))
    assert(train_Y.dtype == torch.long)

    val_X, val_Y = val_loader[0]

    assert(val_X.shape == torch.Size([20, 20, 4]))
    assert(val_X.dtype == torch.float)

    assert(val_Y.shape == torch.Size([20, 20]))
    assert(val_Y.dtype == torch.long)

    for _ in train_loader:
        train_batch_num += 1
    for _ in val_loader:
        val_batch_num += 1

    assert (train_batch_num == (1000-1))
    assert (val_batch_num == (500))  # list not generator


def test_RNN():
    in_dim = 4
    hidden_dim = 100
    out_dim = 1  # binary classification, 0 or 1
    model = RNN(in_dim, hidden_dim, out_dim).to(torch.device)

    X, Y = next(utils.batch_generator(seq_len=10, batch_size=3,
                                      vocab_size=4, device=DEVICE, max_num_batch=100))

    assert (X.shape == torch.Size([10, 3, 4]))
    assert (X.dtype == torch.float32)
    assert (Y.shape == torch.Size([10, 3]))
    assert (Y.dtype == torch.int64)

    logits = model(X)
    assert (logits.shape == torch.Size([10, 3]))
    assert (logits.dtype == torch.float32)

    h_i = torch.zeros([3, 100], device=DEVICE)

    for i in range(10):
        h_i = torch.tanh(X[i, :, :] @ model.W_xh + h_i @ model.W_hh + model.b_h)
        y_i = h_i @ model.W_hy + model.b_y
        assert (torch.allclose(y_i.squeeze_(), logits[i, :]))


def test_num_correct_samples():
    x1 = torch.tensor([[0.5, 0], [-0.5, 3.6], [3.0, 1.5], [-0.2, -2], [1, -0.01]])
    y1 = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    assert (num_correct_samples(x1, y1) == 1)

    x2 = torch.tensor([[0.5, 0], [-0.5, 3.6], [3.0, 1.5], [-0.2, -2], [1, -0.01]])
    y2 = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0], [1, 0]])
    assert (num_correct_samples(x2, y2) == 2)

    x3 = torch.tensor([[0.5, 0], [-0.5, 3.6], [-3.0, 1.5], [-0.2, 2], [1, -0.01]])
    y3 = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0], [1, 0]])
    assert (num_correct_samples(x3, y3) == 0)


def test_validate():
    dataloaders = get_dataloaders(20, 10, train_batch_num=1000, val_batch_num=500)
    train_loader = dataloaders['train']
    val_list = dataloaders['val']

    model = RNN(4, 100, 1).to(DEVICE)
    avg_loss, percent_correct = validate(model, val_list)
    print(avg_loss >= 0)
    print(percent_correct <= 1 and percent_correct >= 0)

def test_train():

    dataloaders = get_dataloaders(20, 7, train_batch_num=10000, val_batch_num=500)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    model = RNN(4, 200, 1).to(DEVICE)

    train(model, train_loader, val_loader)
