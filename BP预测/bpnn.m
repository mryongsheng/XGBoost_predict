% [num]=xlsread("C:\Users\fansen\Desktop/teat_result.csv");
% [m,n]=size(num);
% newNum=[];
% j=1;
% for i = 1 : m
%     if num(i,6)==0 && mod(i,8)==1
%         newNum(j,:)=num(i,:);
%         j=j+1;
%     elseif num(i,6)==1
%         newNum(j,:)=num(i,:);
%         j=j+1;
%     end
% end
% num = newNum;
% P=num(:,2:5)';
% P_train = P(:,1:3000);
% P_test = P(:,3001:3599);
% 
% T=num(:,6)';
% T_train = T(:,1:3000);
% T_test = T(:,3001:3599);
% % ClassType = [0,1];
% % C = [1,2];
% % [P_train,T_train]=SmoteOverSampling(P_train,T_train,ClassType,C,[1,0,1,0],5,'nominal');
% 
% [p1,minp,maxp,t1,mint,maxt]=premnmx(P_train,T_train);
% %��������
% net=newff(minmax(P_train),[4,6,6,6,1],{'tansig','tansig','tansig','tansig','purelin'},'trainlm');
% 
% net.trainParam.epochs=10000;%ѵ����������
% net.trainParam.goal=1e-7;%ѵ��Ŀ������
% net.trainParam.lr=0.01;%ѧϰ������,Ӧ����Ϊ����ֵ��̫����Ȼ���ڿ�ʼ�ӿ������ٶȣ����ٽ���ѵ�ʱ�����������������ʹ�޷�����
% net.trainParam.mc=0.9;%�������ӵ����ã�Ĭ��Ϊ0.9
% net.trainParam.show=25;%��ʾ�ļ������
% 
% 
% [net,tr]=train(net,p1,t1);
% % trainlm, Epoch 0/5000, MSE 0.533351/1e-007, Gradient 18.9079/1e-010
% % trainlm, Epoch 24/5000, MSE 8.81926e-008/1e-007, Gradient 0.0022922/1e-010
% % trainlm, Performance goal met.
%  
%��������
%a=[1.00322344300000,-0.439906879000000,-0.693420373000000,-1.58678798100000,-0.792871807000000,-0.457180767000000,-1.97571679600000,-0.137300744000000,-0.290552486000000,1.79091620400000,-0.0219617190000000]';
%���������ݹ�һ��
arrayt=zeros(1,599);
arrayp=zeros(1,599);
for i =1:599
    a=P_test(:,i);
    a=premnmx(a);
    %���뵽�����������
    b=sim(net,a);
    %���õ������ݷ���һ���õ�Ԥ������
    b=postmnmx(b,mint,maxt);
    
    arrayp(i)=b;
    arrayt(i)=T_test(i);
    if(abs(arrayp(i))<=0.5)
        arrayp(i) = 0;
    else
        arrayp(i) = 1;
    end
end
x_arrix=1:599;
plot(x_arrix,arrayt);
hold on;
plot(x_arrix,arrayp);
% me=mean(abs(arrayp-arrayt));
% legend("��ʵֵ","Ԥ��ֵ")
% title("����ֵ��"+num2str(me));
ZR=0;%0�����ٻ��ʣ���������0/����0
ZP=0;%0����׼ȷ�ʣ���������0�Ե���Ŀ���Լ�������0����Ŀ
FR=0;%1�����ٻ��ʣ���������1/����1
FP=0;%1����׼ȷ�ʣ�
all1=0;%��������1
all0=0;%������0����Ŀ
True1=0;%������1����Ŀ
True0=0;%����0����Ŀ
for i = 1:599
    if arrayt(i) == 0
        True0=True0+1;
    else
        True1=True1+1;
    end
    
    if arrayp(i)==0
        all0=all0+1;
    else
        all1=all1+1;
    end
    if arrayt(i)==0 && arrayp(i)==0
        ZP=ZP+1;
    end
    if arrayt(i)==1 && arrayp(i)==1
        FP = FP+1;
    end
end
ZRr = all0/True0;
FRr = all1/True1;
ZPr = ZP/all0;
FPr = FP/all1; 
    
    