function [BG1]=traindata_cleanV1(a)
% a=xlsread('563-train.xlsx.csv','563-train.xlsx','C100:D13134');
[M,N] = size(a);

AA = isnan(a(:,1));
BB=zeros(M,N);

for i=2:M
    if AA(i,1)~=0
        BB(i,2)=AA(i,1)+BB(i-1,2);
    end
end

for i=M:-1:2
    if AA(i,1)~=0
        BB(i,1)=AA(i,1)+BB(i+1,1);
    end
end


for i=1:M
    if BB(i,2)>6 &&  BB(i,1)>6
        a(i,1)=a(i,2);
    end
end

a1(:,1)=interp1(1:M,a(:,1),1:M,'spline')';


den=fix(M/288);
for i=1:M
    num=mod(i,288);
    for n=1:den-1
        ss(1,n)=a(n*288+num,1);
    end
    a(i,1)=mean(ss(~isnan(ss)));
end

%a2(:,1)=a(:,1);

for i=1:M
    if a1(i,1)~=a(i,1)
        n=min(BB(i,2),BB(i,1));
        a(i,1)=a1(i,1)*0.99^n+a(i,1)-a(i,1)*0.99^n;
        a(i,1)=min(400,a(i,1));
        a(i,1)=max(40,a(i,1));
    end
end

BG1= a(:,1);



%xlswrite('563-train.xlsx.csv',BG1,'563-train.xlsx','AG100:AG13134');
