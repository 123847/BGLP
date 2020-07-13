function [BG1]=testdata_cleanV4(a_test)
[M,N] = size(a_test);%Row and column of data
AA = isnan(a_test(:,1));%The non  is 1, and the non-null is 0
BB=zeros(M,N);%Construct the same size zero matrix

for i=2:M
    if AA(i,1)~=0
        BB(i,2)=AA(i,1)+BB(i-1,2);%ȱʧֵ����
    end
end

for i=M:-1:2
    if AA(i,1)~=0
        BB(i,1)=AA(i,1)+BB(i+1,1); %ȱʧֵ���ȣ���ָ������ĵ�һ��ֵ�ľ��룩
    end
end
% �����￪ʼ��һ��
for i=2:M
    
    if AA(i,1)~=0
        num=mod(i,288);
        den=fix(i/288);
        %if den==0
         %   den=1;
        %end
        if BB(i,2)<4
            a_test(i,1)=2*a_test(i-1)-a_test(i-2); %���Ʋ�ֵ��ȱʧֵ<4����ƽ��ֵ������YSI
        else
            a_test(i,1)=a_test(i,2); %YSI���ȱʧֵ
            AA = isnan(a_test(:,1)); %����ȱʧ��ǩ
            for j=2:M
                if AA(j,1)~=0
                    BB(j,2)=AA(j,1)+BB(j-1,2);%����ȱʧֵ����
                end
            end
            
            
            y1=2*a_test(i-1)-a_test(i-2); %���Ʒ���
            s=mean(a_test(1:i-1)); %�ۼƲ���ֵ�ľ�ֵ
            t=den+1;
            ss(1:t)=s*ones(1,t);
            
            for n=1:den
                r=(n-1)*288+num;
                if r==0
                    r=1;
                end
                if den>=1
                    ss(1,n+den+1)=a_test(r,1);%Blood glucose levels at the same time before the sampling point
                else
                    ss(1,n+den+1)=s;
                end
            end
            
            y2=mean(ss(~isnan(ss))); %��ֵ��ͬһʱ��Ѫ�Ƿ����ľ�ֵ
            
            if 4<=BB(i,2)& BB(i,2)<6
                
                a_test(i,1)=y1*0.999^BB(i,2)+y2-y2*0.999^BB(i,2);%Forgetting factor weighting
                
            else
                
                %a_test(i,1)=y1*0.9^BB(i,2)+y2-y2*0.9^BB(i,2);%�������Ӽ�Ȩ
                a_test(i,1)=y1*0.9^BB(i,2)+y2-y2*0.9^BB(i,2);
                % a_test(i,1)=a_test(i-1,1)
            end
        end
        a_test(i,1)=min(400,a_test(i,1));
        a_test(i,1)=max(40,a_test(i,1));
    end
    
    %%%%%%%%%%
    %����
    %%%%%%%%%%
    
    if AA(i-1,1)~=0 && AA(i,1)==0
        if BB(i-1,2)>=50
            for k=1:42
                yy1(k)=2*a_test(i-k+1)-a_test(i-k+1+1);%First order Taylor series extrapolation
                a_test(i-k,1)=yy1(k)*0.99^k+a_test(i-k,1)-a_test(i-k,1)*0.99^k;
            end
        else if BB(i-1,2)<6
                
                a_test(i,1)=a_test(i,1);
                
                
                
            else
                for k=1:BB(i-1,2)
                    yy1(k)=2*a_test(i-k+1)-a_test(i-k+1+1);%һ��̩�ռ�������
                    a_test(i-k,1)=yy1(k)*0.99^k+a_test(i-k,1)-a_test(i-k,1)*0.99^k;
                    
                end
                
            end
        end
        a_test(i,1)=min(400,a_test(i,1));
        a_test(i,1)=max(40,a_test(i,1));
        
        
    end
    
end



BG1=a_test(:,1);



