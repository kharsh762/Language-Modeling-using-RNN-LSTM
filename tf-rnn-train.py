import tensorflow as tf
import numpy as np
import pickle
import dataset as data
from matplotlib import pyplot as plt
n_hidden = 256
input_size = 1
batch_size = 323
timesteps = 36
learning_rate = 0.001
epoch = 1500
path = 'ptb_train.txt'

def pred(X,n_hidden):
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs,state = tf.nn.static_rnn(cell,X,dtype=tf.float32)
    return outputs

def show_loss(losses):
    x_count=[]
    y_loss=[]
    for i,j in enumerate(losses):
        y_loss.append(i)
        x_count.append(j)
    plt.plot(x_count,y_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN-LSTM Model (Adam Optimizer)')    
    plt.show()

if __name__=="__main__":
    remove_words = ["N","\n" , "<unk>" , "$" ]
    l = data.read_file(path,remove_words)
    word_to_id,id_to_word,vocab_size,l = data.tokenize_file(l)
    x,y = data.build_file(word_to_id,l,vocab_size,one_hot=True)
    
    with open('tf-rnn-save.pickle','wb') as pl:
        pickle.dump([word_to_id,id_to_word],pl)
    
    print("VOCAB SIZE ",vocab_size)
    print("TOTAL WORDS ",len(l))
    losses = []

    X = tf.placeholder(dtype=tf.float32,shape=[None,timesteps,input_size],name="X")
    Y = tf.placeholder(dtype=tf.float32,shape=[None,timesteps,vocab_size])

    out_weights=tf.Variable(tf.random_normal([timesteps,n_hidden,vocab_size]),name="out_weights")
    out_bias=tf.Variable(tf.random_normal([vocab_size]),name="out_bias")

    inp = tf.unstack(X,timesteps,axis=1)
    label = tf.unstack(Y,timesteps,axis=1)

    out = pred(inp,n_hidden)
    logit=tf.matmul(out,out_weights)+out_bias
    prediction = tf.nn.softmax(logit)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=label),name="loss")
    opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction=tf.equal(tf.argmax(prediction,2),tf.argmax(label,2))
    accuracy=tf.multiply(tf.reduce_mean(tf.cast(correct_prediction,tf.float32)),100)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        x = np.reshape(x,(-1,timesteps,input_size))
        y = np.reshape(y,(-1,timesteps,vocab_size))
        i=1
        fdict = {X:x,Y:y}
        while i<epoch:
            _,los,acc = sess.run([opt,loss,accuracy],feed_dict=fdict)
            losses.append(los)
            print("Epoch :" + str(i) + " Loss :" + str(los) + " Accuracy :" + str(acc) + "%")
            i +=1
            if(i%100==0):
                saver.save(sess,"./tf-rnn-model")
        saver.save(sess,"./tf-rnn-model")
    show_loss(losses)
