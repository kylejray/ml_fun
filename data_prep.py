import numpy as np
import pandas as pd
t_year=252

def data_prep():
    dataset=[]  
    ydata=[]
    
    stock_list=['SPY','ATVI','GOOG','IXIC','MS','PCG','PSQ','TM','VGENX','VXO']
    for item in stock_list:
        stock_name=item + '.csv'
        dat=pd.read_csv(stock_name,parse_dates=['Date'])
        price=dat.loc[:,'Adj Close']
        i=0
        while i*t_year < len(price)-504 :
            p=price[i*t_year:(i+1)*t_year]
            p=p.apply(lambda x: x/max(p) )
            dataset.append(np.asarray(p))
            p2=price[(i+2)*t_year]
            p1=price[(i+1)*t_year]
            ydata.append((p2-p1)/p1)
            i=i+1
    
    y=np.asarray(ydata)
    y=np.where(y<0,0,1)
    y=y.astype(np.uint8)
    ND=len(dataset)        
    (x_train,y_train)=(np.asarray(dataset[0:int(ND*.8)]),y[0:int(ND*.8)])
    (x_test,y_test)=(np.asarray(dataset[int(ND*.8):]),y[int(ND*.8):])
            
    return(x_train,y_train,x_test,y_test)
        