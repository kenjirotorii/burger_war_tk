


# def Net(state_size, action_size, hidden_size):
#     inputs = Input(shape=(state_size,))
#     x = Dense(hidden_size, activation = 'relu')(inputs)
#     x = Dense(hidden_size, activation = 'relu')(x)
#     outputs = Dense(action_size, activation = 'linear')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model