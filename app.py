import streamlit as st
import pandas as pd
from data import get_nse_equity, get_company_info, stock_ohlc, get_stock_history, get_symbol_lst
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn import metrics

def write():
    #st.text("(note some companies may not available, we're working on it)")
    df = get_nse_equity()

    symbol_lst = get_symbol_lst()
    new_symbol_lst = [symbol[:-3] for symbol in symbol_lst ]
 
    stock = st.sidebar.text_input("Enter symbol", value="")
    if stock:
        if st.sidebar.button("Choose another company"):
            stock=""
    
    if stock=="":
        st.title("Listed Companies and their Symbols")
        st.text("Tip: Use Ctrl+F for searching")
        
    stock_lst = df
    
    # Check the symbol if it ends with .NS then it's yahoo symbol
    if stock in symbol_lst:
        stock = stock
    elif stock in new_symbol_lst:
        stock = stock + ".NS"

    if stock == "":
        st.table(stock_lst[["NSE SYMBOL","NAME OF COMPANY","YAHOO SYMBOL"]].reset_index(drop=True))
    else:
        company_info = get_company_info(stock)
        st.title(company_info['longName'])
        st.subheader(f"Industry: {company_info['industry']}")
        #Business Summary
        business_information = st.checkbox("Business Summary")
        if business_information:
            st.write(company_info['longBusinessSummary'])
        contact_information = st.checkbox("Contact Info")
        #Company Contact Info
        if contact_information:
            st.write(f"City: {company_info['city']}")
            phone = company_info['phone'].replace(" ","")
            phone = "+"+phone
            st.write(f"Phone: {phone}")
            st.write(f"Website: {company_info['website']}")

        st.sidebar.header("Finance")
        market_info = st.sidebar.checkbox("Market Info")

        if market_info:
            st.subheader("Open - High - Low - Close")
            st.text("1 Month Historical Data")
            time_range = "1mo"
            st.dataframe(stock_ohlc(stock, time_range))
        stock_pred = st.sidebar.checkbox("Stock Price Prediction")
        if stock_pred:
            st.subheader("Next day stock predictions")

            option = {"High":"highest", "Low":"lowest", "Open":"open", "Close":"close"}
            def format_func(item):
                return option[item]
            word = st.selectbox('Select a feature you want to predict', options=list(option.keys()))
            
            num_days= st.text_input("Enter number of days based upon which the model would train", "1")
            num_days = int(num_days)
            st.info(f"Predict the {format_func(word)} Price of the next deal day based on past {num_days} days")

            df = get_stock_history(stock)
            df_new = df[[word]]
            dataset = df_new.values
            #test:30% train:70%
            training_data_len = math.ceil(len(dataset) *.7)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            #Creation of dataset
            train_data = scaled_data[0:training_data_len, : ]
            x_train, y_train = [], []
            for i in range(num_days,training_data_len):
                x_train.append(scaled_data[i-num_days:i,0])
                y_train.append(scaled_data[i,0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

            lstm = Sequential()
            lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
            lstm.add(LSTM(units=50))
            lstm.add(Dense(1))
            lstm.compile(loss='mean_squared_error', optimizer='adam')

            if st.button('Train the model!'):
                with st.spinner("Training may take time based on number of days. Please wait..."):
                    history_lstm = lstm.fit(x_train, y_train, epochs=25, batch_size=10, verbose=2)
                    st.info("Model is ready!")
                st.button("Re-run")

                test_data = scaled_data[training_data_len - num_days: , : ]
                x_test = []
                y_test =  dataset[training_data_len : , : ]
                for i in range(num_days,len(test_data)):
                    x_test.append(test_data[i-num_days:i,0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

                pred = lstm.predict(x_test)
                pred = scaler.inverse_transform(pred)

                st.header("Prediction")
                inputs = df_new.iloc[len(df_new)-num_days:len(df_new)].values
                inputs = inputs.reshape(-1,1)
                inputs  = scaler.fit_transform(inputs)
                p = inputs.reshape(1,num_days,1)
                result = lstm.predict(p)
                result = scaler.inverse_transform(result)
                st.write("The {} price of next deal day is :".format(word))
                st.success(round(float(result),2))

                st.header("Model Evaluation")
                st.write('R Square - ',round(metrics.r2_score(y_test,pred),2))
                
