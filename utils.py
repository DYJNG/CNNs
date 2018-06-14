from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
import datetime
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def train(net, train_data, valid_data, ctx, num_epoches, optimizer='adam', 
          lr=0.01, lr_decay=0.1, lr_period=50, momentum=0.9, weight_decay=0, 
          cost_peroid=10, print_cost=False):
    if optimizer == 'momentum':
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 
                                                              'momentum': momentum, 
                                                              'wd': weight_decay})
    elif optimizer == 'adam':
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 
                                                               'wd': weight_decay})
    
    train_costs = []
    valid_costs = []
    v_loss_train = 0
    n_iter_train = 0
#     v_loss_valid = 0
#     n_iter_valid = 0
    for epoch in range(num_epoches):
        pre_time = datetime.datetime.now()
        train_acc = 0
#         train_loss = 0
        if (epoch+1) % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.astype('float32').as_in_context(ctx)
            with ag.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_acc += nd.mean(output.argmax(axis=1) == label).asscalar()
#             train_loss += nd.mean(loss).asscalar()
            cur_loss = nd.mean(loss).asscalar()
            v_loss_train = 0.9 * v_loss_train + 0.1 * cur_loss
            n_iter_train += 1
            corr_loss_train = v_loss_train / (1 - pow(0.9, n_iter_train))
            
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - pre_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time %02d:%02d:`%02d, ' % (h, m, s)
        
        if valid_data is not None:
            valid_acc = 0
            valid_loss = 0
            for data, label in valid_data:
                data = data.as_in_context(ctx)
                label = label.astype('float32').as_in_context(ctx)
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                valid_acc += nd.mean(output.argmax(axis=1) == label).asscalar()
#                 cur_loss = nd.mean(loss).asscalar()
#                 v_loss_valid = 0.9 * v_loss_valid + 0.1 * cur_loss
#                 n_iter_valid += 1
#                 corr_loss_valid = v_loss_valid / (1 - pow(0.9, n_iter_valid))
                valid_loss += nd.mean(loss).asscalar()
            epoch_str = 'Epoch %d, Train_loss: %s, Train_acc: %s, Valid_acc: %s, ' % (epoch+1, 
                                                                                    corr_loss_train, 
                                                                                    train_acc/len(train_data), 
                                                                                    valid_acc/len(valid_data))
        else:
            epoch_str = 'Epoch %d, Train_loss: %s, Train_acc: %s, ' % (epoch+1, 
                                                                     corr_loss_train, 
                                                                     train_acc/len(train_data))
        if print_cost and (epoch+1) % cost_peroid == 0:
            train_costs.append(corr_loss_train)
#             train_costs.append(train_loss/len(train_data))
            valid_costs.append(valid_loss/len(valid_data))
        
        print(epoch_str + time_str + 'lr: %f' % trainer.learning_rate)
        
    if print_cost:
        x_axis = np.linspace(0, num_epoches, len(train_costs), endpoint=True)
        plt.semilogy(x_axis, train_costs)
        plt.semilogy(x_axis, valid_costs)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()