%% Data cleaning of training set;Draw raw and processed data
a_train=xlsread('540-train.xlsx.csv','540-train.xlsx','C142:D13250');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'540_train.fig');
xlswrite('540-train.xlsx.csv',BG1,'540-train.xlsx','AG142:AG13250');

a_train=xlsread('544-train.xlsx.csv','544-train.xlsx','C3:D12673');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'544_train.fig');
xlswrite('544-train.xlsx.csv',BG1,'544-train.xlsx','AG3:AG12673');

a_train=xlsread('552-train.xlsx.csv','552-train.xlsx','C138:D11234');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b'); hold off;
saveas(figure,'552_train.fig');
xlswrite('552-train.xlsx.csv',BG1,'552-train.xlsx','AG138:AG11234');

a_train=xlsread('563-train.xlsx.csv','563-train.xlsx','C37:D13134');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'563_train.fig');
xlswrite('563-train.xlsx.csv',BG1,'563-train.xlsx','AG37:AG13134');

a_train=xlsread('567-train.xlsx.csv','567-train.xlsx','C3:D13538');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'567_train.fig');
xlswrite('567-train.xlsx.csv',BG1,'567-train.xlsx','AG3:AG13538');

a_train=xlsread('584-train.xlsx.csv','584-train.xlsx','C3:D13250');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'584_train.fig');
xlswrite('584-train.xlsx.csv',BG1,'584-train.xlsx','AG3:AG13250');

a_train=xlsread('596-train.xlsx.csv','596-train.xlsx','C661:D14290');
[BG1]=traindata_cleanV1(a_train);
figure=plot(BG1,'r');hold on;plot(a_train(:,1),'b');hold off;
saveas(figure,'596_train.fig');
xlswrite('596-train.xlsx.csv',BG1,'596-train.xlsx','AG661:AG14290');