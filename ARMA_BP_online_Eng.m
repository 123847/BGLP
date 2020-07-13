%% Empty the environment 
clc;
clear;
close all;
tic
%% Import training set and test set, and organize the data
[train_num]=xlsread('C:\Users\MN\Desktop\traindataV3\552-train.xlsx.csv');
[test_num]=xlsread('C:\Users\MN\Desktop\testdataV4\552-test.xlsx.csv');
trainNum=[train_num(:,2),train_num(:,32)];  %Real BG, supplement BG
trainNum(find(isnan(trainNum)==1)) = 0 ; %NAN transfer 0
trainNum(trainNum(:,2)==0, :) = [];  %The addition of BG contains 0 rows
%补充的BG用于训练
train_BG=trainNum(:,2);
train_BG_long=length(train_BG); 
train_number=train_BG(:,1);  
train_number_long=length(train_number);  %The record length used to start recording the predicted value
%Used to calculate the true BG of the error
train_true_BG=trainNum(:,1); 
%% Limit[40,400]
Max=max(train_BG); 
Min=min(train_BG);
Limit_min=40;
Limit_max=400;
%% The stationarity test was carried out on the training set. 
%When it was 1, 0, there was no correlation yd1_h_adf =1，yd1_h_kpss =0，
Y=train_BG; 
y_h_adf = adftest(Y)
y_h_kpss = kpsstest(Y)
aimY = diff(Y); %A first order differential
y_h_adf = adftest(aimY)
y_h_kpss = kpsstest(aimY)
%% Drawing, AIC, BIC determine the model order Q, P
figure(1) %The autocorrelation and partial autocorrelation of the original data are plotted on a graph
subplot(2,1,1)
autocorr(aimY)
subplot(2,1,2)
parcorr(aimY)
adf=adftest(aimY); %if adf==1，it is a stationary time series
%AIC、BIC
logl = zeros(10,1);
P = zeros(10,1);
for p = 1:10   
    mod = arima('ARLags',1:p); %The AR model was constructed, with a lag of P
        [fit,~,logl(p)] = estimate(mod,aimY,'display','off'); % Estimated parameters
    P(p) = p;  
end
[aic_p,bic_p] = aicbic(logl,P+1,train_number_long-1); %AIC、BIC
BICmin_p = min(min(bic_p));
AICmin_p = min(min(aic_p));
[row1,column1] = find(bic_p == BICmin_p);   %The lowest p of BIC     
[row2,column2] = find(aic_p == AICmin_p);   %The lowest p of AIC 
log2 = zeros(10,1);
Q = zeros(10,1);
for q = 1:10   
    mod = arima('MALags',1:q); %The MA model was constructed, with a lag of q
        [fit,~,log2(q)] = estimate(mod,aimY,'display','off'); 
    Q(q) = q;  
end
[aic_q,bic_q] = aicbic(log2,Q+1,train_number_long-1);
BICmin_q = min(min(bic_q));
AICmin_q = min(min(aic_q));
[row3,column3] = find(bic_q == BICmin_q);         
[row4,column4] = find(aic_q == AICmin_q);    
disp(['BIC_PQ：',num2str(row1),'；',num2str(row3)]);
disp(['AIC_PQ：',num2str(row2),'；',num2str(row4)]);
%% Residual test
Mdl = arima(row2, 1, row4);  %variable P、Q、I
EstMdl = estimate(Mdl,Y);
[res,~,logL] = infer(EstMdl,Y);   %res
diffRes0 = diff(res);  
SSE0 = res'*res;
DW0 = (diffRes0'*diffRes0)/SSE0;  %When the value is close to 2, the sequence can be considered to have no first-order correlation
disp(['DW0：',num2str(DW0)]);
%% Determine each input value
train_p=input('Input order P='); %P,Q in ARMA
train_q=input('Input order Q=');
window=input('Enter ARMA slide window='); %Window size;
Patient_ID=input('Patients with number='); %Patients with number;
Thirty=input('Input 0 or 1='); %30min
Sixty=input('Input 0 or 1=');  %60min
n=input('Number of input neural networks='); %The actual number is equal to n+1
m=input('Number of hidden neurons='); %The actual number is equal to m
train_use=input('train_use=');  %The size required for the initial test set
Variable_n=input('Input calculation correlation=');  %The true number is equal to the input value plus 1
update_window=input('Error model slide window='); %The actual number is equal to UPDATE
update_number=input('Model data update='); %The actual number is equal to UPDATE
%% It is used to select a small number of training sets to obtain the prediction error
train_part=input('Enter the training set size='); % Window size + error model slider size
%Supplementary BG for training
train_number=train_BG(train_number_long-(train_part-1):train_number_long,1);  
train_number_long=length(train_number);  %记录长度，用于开始记录预测值使用
%Training set training
%Test set storage space
train_Ym=zeros(train_number_long-6*Thirty-12*Sixty,1);   
for train_k=window:train_number_long   
%% Model fitting prediction  
    train_X=train_number(train_k-window+1:train_k,1);   %Fetch window size data
    train_y = diff(train_X);     %Differential treatment
    train_ToEstMd = arima('ARLags',train_p,'MALags',train_q,'Constant',0);%Specifies the structure of the model
    [train_EstMd,EstParamCov,LogL,info] = estimate(train_ToEstMd,train_y);%The model fitting 
    train_w_Forecast = forecast(train_EstMd,6*Thirty+12*Sixty,'Y0',train_y);  %30min：6；60min：12
    %The prediction function is obtained and the data is predicted
    %The difference values are restored and recorded
    train_yhat = train_X(end) + cumsum(train_w_Forecast); 
    
    train_Yn=train_yhat(6*Thirty+12*Sixty,1);   %Extract predicted value
%% Limit the size   
    if train_Yn<=Limit_min
         train_Yn=Limit_min;
    end
    if train_Yn>=Limit_max
         train_Yn=Limit_max;
    end
%% The final training set predicts the result
    train_Ym(train_k+6*Thirty+12*Sixty,1)=train_Yn;  
end
%% Extract the predicted results of the training set
train_Ym_long=length(train_Ym);
train_Ypredict=train_Ym(1:train_Ym_long-6*Thirty-12*Sixty,1); %Record the predicted values of the training set
train_true_BGlong=length(train_true_BG);
%% The error is calculated by removing the null value of the observed value  
%Populating data is used for comparison
train_Yall=[train_Ypredict,train_number]; 
train_YallLong=length(train_Yall);  % The total length

train_Y_all=train_Yall(1:train_YallLong,1:2);  %Predictive value.The real value
train_Y_all1=train_Y_all(train_Y_all(:,1)~=0,:);  
train_Y_all2=train_Y_all1(train_Y_all1(:,2)~=0,:);  
train_Ylast=train_Y_all2;
train_sumLong=length(train_Ylast);  %Find the total length of the remaining training set
%Training set error storage space
train_Error=zeros(train_sumLong,1);  
for train_j=1:train_sumLong
    train_Error(train_j,1)=train_Ylast(train_j,1)-train_Ylast(train_j,2);  
end
%Drawing and analyzing the training set error
figure(2)
plot(train_Error)
%% Correlation analysis of training set error true value
Matrix_true=zeros(Variable_n+1,train_sumLong-Variable_n-(6*Thirty+12*Sixty));
Matrix_error=zeros(Variable_n+1,train_sumLong-Variable_n-(6*Thirty+12*Sixty));
Matrix_target=zeros(1,train_sumLong-Variable_n-(6*Thirty+12*Sixty));
Relevance_true=zeros(1,6);
Relevance_error=zeros(1,6);
for Matrix_i=1:train_sumLong-Variable_n-(6*Thirty+12*Sixty) 
    Matrix_j=Matrix_i;
    Matrix_true(:,Matrix_i)=train_Ylast(Matrix_j:Matrix_j+Variable_n,2);  
    Matrix_error(:,Matrix_i)=train_Error(Matrix_j:Matrix_j+Variable_n,1); 
    Matrix_target(:,Matrix_i)=train_Error(Matrix_j+Variable_n+6*Thirty+12*Sixty,1);  
end
%Find the correlation for each column in the matrix
for Relevance_i=1:Variable_n+1
    
    Relevance_X_true=Matrix_true(Relevance_i,:); 
    Relevance_X_error=Matrix_error(Relevance_i,:);
    
    Relevance_X_true=Relevance_X_true';
    Relevance_X_error=Relevance_X_error';
    Relevance_Y=Matrix_target';
    
    Relevance_true(1,Relevance_i)=corr(Relevance_X_true,Relevance_Y,'type','pearson');
    Relevance_error(1,Relevance_i)=corr(Relevance_X_error,Relevance_Y,'type','pearson');        
end
%Final correlation table, from lag 5 steps to current value, before: observed value, after: error
Relevance_Matrix=[Relevance_true,Relevance_error]; 
% Correlation coefficient 0.00-±0.3 is micro-weight correlation 
%±0.30-±0.50 is real correlation, ±0.50-±0.80 is significant correlation, ±0.80-±1.00 is highly correlation
%Select correlation greater than 0.5
Relevance_number=find(Relevance_Matrix>=0.3 | Relevance_Matrix<=-0.3);
disp(['Relevance_sign: ',num2str(Relevance_number),'；''Variables_Number: ',num2str(length(Relevance_number))]);
%% Test sets and data collation
test_BG=[test_num(:,2),test_num(:,32)]; %Extract CGM and supplement CGM
test_BG(find(isnan(test_BG)==1)) = 0 ; %NAN with 0
test_BG(test_BG(:,2)==0, :) = [];  %Divide by CGM and you have 0 rows
test_number=test_BG(:,2); 
test_number_long=length(test_number);
%Used to calculate the true BG of the error
test_true_BG=test_BG(:,1); 
%% Total data set
%According to the window size, when all the data of the training set is carried out
test_use_train=train_BG(train_BG_long-train_use+1:train_BG_long,1);
train_number_long=length(test_use_train);
all_number=[test_use_train;test_number];
all_number_long=length(all_number);
%% Test set training model
%% Define storage space
de=0;
test_Ypredict=zeros(all_number_long,1);
Y_de=zeros(test_number_long,3);
test_Y_last=zeros(test_number_long,3);
test_true_new=zeros(test_number_long,1);
test_error=zeros(test_number_long,1);
%% Required inputs for the prediction set
%True value part
train_true_part(:,1)=train_Ylast(train_sumLong-n-5*Thirty-11*Sixty:train_sumLong,2);
%True values and errors are divided into two columns
train_true_error_part=zeros(n+1+5*Thirty+11*Sixty,2);
train_true_error_part(:,1)=train_Ylast(train_sumLong-n-5*Thirty-11*Sixty:train_sumLong,2);
train_true_error_part(1:1+5*Thirty+11*Sixty,2)=train_Error(train_sumLong-5*Thirty-11*Sixty:train_sumLong,1); 
%The error part
train_error_part(:,1)=train_Error(train_sumLong-n-5*Thirty-11*Sixty:train_sumLong,1); 
%% Training network;The sliding window training model is adopted
for test_k=window:all_number_long   
%% Training set model online training and prediction
    test_X=all_number(test_k-window+1:test_k,1);  
    test_y = diff(test_X);    
    test_ToEstMd = arima('ARLags',train_p,'MALags',train_q,'Constant',0);
    [test_EstMd,EstParamCov,LogL,info] = estimate(test_ToEstMd,test_y);
    %提取训练集预测值
    if test_k>=train_number_long-5*Thirty-11*Sixty  %30min-5;60min-11；      
        test_w_Forecast = forecast(test_EstMd,6*Thirty+12*Sixty,'Y0',test_y);              
        test_yhat = test_X(end) + cumsum(test_w_Forecast);
        Ypredict= test_yhat(6*Thirty+12*Sixty,1); 
%% Limit the size   
        if Ypredict<=Limit_min
           Ypredict=Limit_min;
        end
        if Ypredict>=Limit_max
           Ypredict=Limit_max;
        end
        test_Ypredict(test_k+6*Thirty+12*Sixty,1,1)=Ypredict;
%% Neural network input, update
        %Storage space Settings
        All_input_true=zeros(n+1,update_window);
        All_input_true_error=zeros(n+1+1,update_window);
        All_input_error=zeros(n+1,update_window);
        All_target_error=zeros(1,update_window);
        %Extract neural network training input
        AllError_i=0;
        %Model sliding window learning update       
        for AllError_j=train_sumLong-n-(6*Thirty+12*Sixty)+de-(update_window-1):train_sumLong-n-(6*Thirty+12*Sixty)+de  
            AllError_i=1+AllError_i;
            All_Ylast=[train_Ylast(:,2);test_true_new(:,1)]; %The real value
            All_Error=[train_Error;test_error];   %error
            All_input_true(:,AllError_i)=All_Ylast(AllError_j:AllError_j+n,1);  
            All_input_true_error(:,AllError_i)=[All_Ylast(AllError_j:AllError_j+n,1);All_Error(AllError_j+n,1)]; 
            All_input_error(:,AllError_i)=All_Error(AllError_j:AllError_j+n,1); 
            All_target_error(:,AllError_i)=All_Error(AllError_j+n+6*Thirty+12*Sixty,1); 
        end               
%% Error model training 
        %Model update rule
        if rem(de,update_number)==0  
           s=1;
        else
           s=0;
        end
        if s==1
            %Unified use
           [train_target,ps_target]=mapminmax([All_target_error]); 
            LP.lr=0.000001;%Learning rate
           % Real value substitution
            [All_input_true,ps_true]=mapminmax([All_input_true]);%The normalized
            net_true=newff(All_input_true,train_target,m,{'tansig','purelin'},'trainlm');
            net_true.trainParam.max_fail = 6; 
            net_true.trainParam.epochs=1000;
            net_true.trainParam.goal=0.000001;
            
            net_true.divideParam.trainRatio = 70/100;
            net_true.divideParam.valRatio = 15/100;
            net_true.divideParam.testRatio = 15/100;
            net_true=train(net_true,All_input_true,train_target);  

           % The real value and the current error
            [All_input_true_error,ps_true_error]=mapminmax([All_input_true_error]);
            net_true_error=newff(All_input_true_error,train_target,m,{'tansig','purelin'},'trainlm');
            net_true_error.trainParam.max_fail = 6;
            net_true_error.trainParam.epochs=1000;
            net_true_error.trainParam.goal=0.000001;
            
            net_true_error.divideParam.trainRatio = 70/100;
            net_true_error.divideParam.valRatio = 15/100;
            net_true_error.divideParam.testRatio = 15/100;
            net_true_error=train(net_true_error,All_input_true_error,train_target);  %训练模型

            %Error in
            [All_input_error,ps_error]=mapminmax([All_input_error]);
            net_error=newff(All_input_error,train_target,m,{'tansig','purelin'},'trainlm');
            net_error.trainParam.max_fail = 6; 
            net_error.trainParam.epochs=1000; 
            net_error.trainParam.goal=0.000001;
           
            net_error.divideParam.trainRatio = 70/100;
            net_error.divideParam.valRatio = 15/100;
            net_error.divideParam.testRatio = 15/100;
            net_error=train(net_error,All_input_error,train_target); 
        end
%% Calculated prediction error
        %Make data storage label, record from test set 1
        de = test_k-(train_number_long-5*Thirty-11*Sixty)+1;  
        %Total true value and update the true value
        All_true=[train_true_part;test_true_new];  
        input_true_GYQ=[All_true(de:de+n,1)];
        input_true=mapminmax('apply',input_true_GYQ,ps_true);   %Apply the seed normalization before
        output_true=net_true(input_true);  %Make the prediction and set the input
        output_true_FGY=mapminmax('reverse',output_true,ps_target);
        %Total true value and error, and update the true value and error
        All_true_error(:,1)=[train_true_error_part(:,1);test_true_new];
        All_true_error(:,2)=[train_true_error_part(:,2);test_error];   
        input_true_error_GYQ=[All_true_error(de:de+n,1);All_true_error(de,2)];
        input_true_error=mapminmax('apply',input_true_error_GYQ,ps_true_error);   
        output_true_error=net_true_error(input_true_error);  
        output_true_error_FGY=mapminmax('reverse',output_true_error,ps_target);
        %Total error and update to the error
        All_error=[train_error_part;test_error];  
        input_error_GYQ=[All_error(de:de+n,1)];
        input_error=mapminmax('apply',input_error_GYQ,ps_error);   
        output_error=net_error(input_error);  
        output_error_FGY=mapminmax('reverse',output_error,ps_target);
        %Record output prediction errors in three columns
        Y_de(de,1)=output_true_FGY;  
        Y_de(de,2)=output_true_error_FGY; 
        Y_de(de,3)=output_error_FGY; 
%% Limit the prediction error[-50,50]
        if Patient_ID~=567 ||  Sixty ~=1
           for De_Limit=1:3
               if  Y_de(de,De_Limit)>=50
                   Y_de(de,De_Limit)=50;
               end
               if  Y_de(de,De_Limit)<=-50
                   Y_de(de,De_Limit)=-50;
               end
           end
        end
        true_test_Ypredict=test_Ypredict(train_number_long+1:all_number_long,1);
        test_Y_de(1,1)=true_test_Ypredict(de,1)+(-Y_de(de,1)); %The prediction error is reversed and the final true value is obtained
        test_Y_de(1,2)=true_test_Ypredict(de,1)+(-Y_de(de,2)); 
        test_Y_de(1,3)=true_test_Ypredict(de,1)+(-Y_de(de,3)); 
%% The final predictive value limit[40,400]
        for testPredict=1:3
            if test_Y_de(1,testPredict)>=400
               test_Y_de(1,testPredict)=400;
            end
            if test_Y_de(1,testPredict)<=40
               test_Y_de(1,testPredict)=40;
            end
        %Three final predictions were recorded
            test_Y_last(de,testPredict)=test_Y_de(1,testPredict);
        end    
%% Calculate the newly achieved errors, add them to the training set, and update the model
        test_error(de,1)=true_test_Ypredict(de,1)-test_number(de,1);
        test_true_new(de,1)=test_number(de,1);
    end
    if de>=test_number_long
       break
    end
end
%Draw the error and forecast error respectively
figure(3)
subplot(3,1,1)
plot(test_error(:,1))
hold on
plot(Y_de(:,1))
legend('Test set error','Model 1 prediction')
subplot(3,1,2)
plot(test_error(:,1))
hold on 
plot(Y_de(:,2))
legend('Test set error','Model 2 prediction')
subplot(3,1,3)
plot(test_error(:,1))
hold on 
plot(Y_de(:,3))
legend('Test set error','Model 3 prediction')
suptitle('Test set real error and prediction error comparison diagram')
%% Extract the predicted value of the final test set
Y_predict=test_Y_last;  %
Y_BGtrue=test_true_BG(:,1); %Truth 
%% Calculate RMSE and MAE
%% Index 1: Null value calculation without observation value
Yall_1=[Y_predict,test_Ypredict(train_number_long+1:all_number_long),Y_BGtrue];
YallLong_1=length(Yall_1); 
%Extract from the 13th prediction value
Y1_all=Yall_1(13:YallLong_1,1:5);
%Get rid of the true value 0
Y_all_1=Y1_all(Y1_all(:,1)~=0,:);  
Y_all_2=Y_all_1(Y_all_1(:,5)~=0,:);   
Y1_last=Y_all_2;
sumLong_1=length(Y1_last);  %Find the total length of the rest
Error_1=zeros(sumLong_1,3);
Error_1_No=zeros(sumLong_1,1);
for j=1:sumLong_1
    Error_1(j,1)=Y1_last(j,1)-Y1_last(j,5);  %Separate error
    Error_1(j,2)=Y1_last(j,2)-Y1_last(j,5);
    Error_1(j,3)=Y1_last(j,3)-Y1_last(j,5);
    Error_1_No(j,1)=Y1_last(j,4)-Y1_last(j,5);
end
%Sum of error square and RMSE;Sum the absolute value of error and MAE
errorSum_1=zeros(1,3);
RMSE_1=zeros(1,3);
errorAbs_1=zeros(1,3);
MAE_1=zeros(1,3);
for Evaluate_1=1:3
    errorSum_1(1,Evaluate_1)=sum(Error_1(:,Evaluate_1).*Error_1(:,Evaluate_1));
    RMSE_1(1,Evaluate_1)= sqrt(errorSum_1(1,Evaluate_1)/sumLong_1);
    errorAbs_1(1,Evaluate_1)=sum(abs(Error_1(:,Evaluate_1))); 
    MAE_1(1,Evaluate_1)=errorAbs_1(1,Evaluate_1)/sumLong_1; 
end
errorSum_1_No=sum(Error_1_No.*Error_1_No);  
RMSE_1_No= sqrt(errorSum_1_No/sumLong_1); 
errorAbs_1_No=sum(abs(Error_1_No)); 
MAE_1_No=errorAbs_1_No/sumLong_1;
%Find the minimum and display
disp(['IndexOne_RMSE_No：',num2str(RMSE_1_No)]);
disp(['IndexOne_MAE_No：',num2str(MAE_1_No)]);

MinIndexOne_RMSE= min(min(RMSE_1));
MinIndexOne_MAE= min(min(MAE_1));
[Method_x,Method_y]= find(RMSE_1==min(min(RMSE_1)));
disp(['Method: ',num2str(Method_y)]);
disp(['IndexOne_RMSE：',num2str(MinIndexOne_RMSE)]);
disp(['IndexOne_MAE：',num2str(MinIndexOne_MAE)]);
%% Let me draw a graph of index one
figure(4)
x=1:1:test_number_long;
plot(x,Yall_1(:,Method_y),'b');
hold on
plot(x,Yall_1(:,5),'y');
hold on 
plot(x,Yall_1(:,4),'g');
legend('Fill in the residual data','Real data','Uncompensated residual data');
title('A comparison of the patients final predicted values')
