%% Test set missing data population
a_test1=xlsread('540-test.xlsx.csv','540-test.xlsx','C3:D3067');
[BG1]=testdata_cleanV2(a_test1);%Limit12
[BG2]=testdata_cleanV4(a_test1);%Limit6
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'540_6steps.fig');
xlswrite('540-test.xlsx.csv',BG1,'540-test.xlsx','AG3:AG3067'); %Limit12
xlswrite('540-test.xlsx.csv',BG2,'540-test.xlsx','AH3:AH3067'); %Limit6

a_test1=xlsread('544-test.xlsx.csv','544-test.xlsx','C4:D3140');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'544_6steps.fig');
xlswrite('544-test.xlsx.csv',BG1,'544-test.xlsx','AG4:AG3140');
xlswrite('544-test.xlsx.csv',BG2,'544-test.xlsx','AH4:AH3140');

a_test1=xlsread('552-test.xlsx.csv','552-test.xlsx','C3:D3952');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'552_6steps.fig');
xlswrite('552-test.xlsx.csv',BG1,'552-test.xlsx','AG3:AG3952');
xlswrite('552-test.xlsx.csv',BG2,'552-test.xlsx','AH3:AH3952');

a_test1=xlsread('563-test.xlsx.csv','563-test.xlsx','C6:D2696');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'563_6steps.fig');
xlswrite('563-test.xlsx.csv',BG1,'563-test.xlsx','AG6:AG2696');
xlswrite('563-test.xlsx.csv',BG2,'563-test.xlsx','AH6:AH2696');

a_test1=xlsread('567-test.xlsx.csv','567-test.xlsx','C3:D2873');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'567_6steps.fig');
xlswrite('567-test.xlsx.csv',BG1,'567-test.xlsx','AG3:AG2873');
xlswrite('567-test.xlsx.csv',BG2,'567-test.xlsx','AH3:AH2873');

a_test1=xlsread('584-test.xlsx.csv','584-test.xlsx','C3:D2997');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'584_6steps.fig');
xlswrite('584-test.xlsx.csv',BG1,'584-test.xlsx','AG3:AG2997');
xlswrite('584-test.xlsx.csv',BG2,'584-test.xlsx','AH3:AH2997');

a_test1=xlsread('596-test.xlsx.csv','596-test.xlsx','C19:D3021');
[BG1]=testdata_cleanV2(a_test1);
[BG2]=testdata_cleanV4(a_test1);
figure=plot(BG1,'r');hold on;plot(BG2,'g');hold on;plot(a_test1(:,1),'b');hold off;
saveas(figure,'596_6steps.fig');
xlswrite('596-test.xlsx.csv',BG1,'596-test.xlsx','AG19:AG3021');
xlswrite('596-test.xlsx.csv',BG2,'596-test.xlsx','AH19:AH3021');