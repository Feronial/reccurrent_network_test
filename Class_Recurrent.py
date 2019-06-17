class recurrent_network():

    def __init__(self, train_X, train_Y, test_X, test_Y, neur_size):
        
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.neur_size = neur_size
        self.network_types = {'lstm' : LSTM(neur_size, activation = 'relu'), 
                              'rnn' : SimpleRNN(neur_size, activation = 'relu'), 
                              'gru' : GRU(neur_size, activation = 'relu')}

    
    def build_net(self, network_type, epoch = 300, batch_size = 6):
        
        print(train_X.shape[1],train_X.shape[2])
        
        inputs = Input(shape = (train_X.shape[1],train_X.shape[2]))

        layer_1 = self.network_types[network_type](inputs)
        
        batch_norm = BatchNormalization()(layer_1)
        
        drop_1 = Dropout(0.25)(batch_norm)
        
        dense_1 = Dense(16, activation='relu')(drop_1)

        outputs = Dense(1, activation= 'softplus')(layer_1)

        model = Model(inputs = inputs, outputs = outputs)
        
        model.compile(loss='mean_absolute_percentage_error', 
              metrics=['mae'],
              optimizer='Adam', 
              )
        
        model.summary()
        
        history = model.fit(x = self.train_X,
                            y = self.train_Y,
                            batch_size = batch_size,
                            epochs = epoch,
                            validation_data= (test_X,test_Y))
        
        return model, history
        
        
    def build_all(self, index):
        
        self.result_df_list = []
        
        for net_type in self.network_types.keys():
            
            model, history = self.build_net(net_type)
            self.eval_model(model, history, index)
            
            
        return 0    
            
    def eval_model(self,model, history, index, verbose = 0):
        
        pred = model.predict(self.test_X)
        
        part1 = str(model.layers[1]).split(' ')[0]
        model_type = part1.split('.')[3]
        
        result_df = pd.DataFrame(columns = ['YILAY', 'ACT', 'PRED', 'ERROR','ERROR_SEG','ABS_TOTAL'])
        result_df['YILAY'] = index
        result_df.loc[:, 'ACT'] = self.test_Y
        result_df.loc[:, 'PRED'] = pred
        result_df.loc[:, 'ERROR'] = abs(result_df['ACT'] / result_df['PRED'] - 1)
        result_df.loc[:, 'ERROR_SEG'] = result_df['ERROR'].apply(self.segmentor)
        result_df.loc[0, 'ABS_TOTAL'] = np.mean(abs(result_df['ACT'] / result_df['PRED'] - 1))
        
        if verbose == 1:
            
            print(result_df)
            
            result_df.to_excel( str(model_type) + 
                          '_' + str(datetime.datetime.now().minute) +
                          '_' + str(datetime.datetime.now().day) + 
                          '_' + str(datetime.datetime.now().month) +
                          '_' + str(datetime.datetime.now().year) + '.xlsx')
            
            self.print_graphic(history)
            self.result_df_list.append(result_df)

        
        else:
            
            result_df.to_excel( str(model_type) + 
                          '_' + str(datetime.datetime.now().minute) +
                          '_' + str(datetime.datetime.now().day) + 
                          '_' + str(datetime.datetime.now().month) +
                          '_' + str(datetime.datetime.now().year) + '.xlsx')
                
            self.print_graphic(history)
            self.result_df_list.append(result_df)

        
    def save_rec_model(self, model, name, format_type = 'excel'):
            
        save_model(model, name + '.h5')
        
        return 0
    
    def segmentor(self, x):
    
        if x <= 0.05:
            return 'SEG_5'

        elif x > 0.05 and x <= 0.1:
            return 'SEG_5_10'

        elif x > 0.1 and x <= 0.15:
            return 'SEG_10_15'

        elif x > 0.15 and x <= 0.20:
            return 'SEG_15_20'

        else:
            return 'SEG_20+'

    
    def print_graphic(self, history):
        
        plt.plot(history.history['mean_absolute_error'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
    def model_comperison(self):
        
        print('LSTM --- ---- ---- ---- --- --- ---- ---- ---- --- --- ---- ---- ---- ---')
        print(modeller.result_df_list[0])
        print('RNN --- ---- ---- ---- --- --- ---- ---- ---- --- --- ---- ---- ---- ---')
        print(modeller.result_df_list[1])
        print('GRU --- ---- ---- ---- --- --- ---- ---- ---- --- --- ---- ---- ---- ---')
        print(modeller.result_df_list[2])
