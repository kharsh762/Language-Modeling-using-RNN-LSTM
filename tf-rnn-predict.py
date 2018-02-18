import tensorflow as tf 
import numpy as np 
import pickle
import dataset as data

n_hidden=256
meta_fname='tf-rnn-model.meta'
fname='ptb_test.txt'
remove_words= ["N","\n" , "<unk>" , "$" ]
timesteps=36
input_size = 1
batch_size=3


def predict(X,n_hidden):
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs,state = tf.nn.static_rnn(cell,X,dtype=tf.float32)
    return outputs

def restore(meta_fname):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_fname)
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        X = tf.placeholder(dtype=tf.float32,shape=[None,timesteps,input_size])
        out_weights = graph.get_tensor_by_name('out_weights:0') 
        out_bias = graph.get_tensor_by_name('out_bias:0')
        return X,out_weights,out_bias

def next_word(X,out_weights,out_bias):      
    inp = tf.unstack(X,timesteps,axis=1)
    out = predict(inp,n_hidden)
    logit=tf.matmul(out,out_weights)+out_bias
    prediction = tf.nn.softmax(logit)
    ans = tf.argmax(prediction,2)
    ans = tf.transpose(ans)
    return ans
 
if __name__=='__main__':
    try:
        with open('tf-rnn-save.pickle','rb') as pl:
            word_to_id,id_to_word = pickle.load(pl)
    except:
        print("No pickle file found !")
        exit()
    remove_words = ["N","\n" , "<unk>" , "$" ]            
    l = data.read_file(fname,remove_words)
    print("Total words :")
    print(l)
    x,_ = data.build_file(word_to_id,l,one_hot=False)
    X,out_weights,out_bias = restore(meta_fname)
    next_ans = next_word(X,out_weights,out_bias)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x = np.reshape(x,(-1,timesteps,input_size))
        final_word = sess.run(next_ans,feed_dict={X:x})
        for i in range(batch_size):         
            print('OUTPUT')
            print(id_to_word[final_word[i][-1]])