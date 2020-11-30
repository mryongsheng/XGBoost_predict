
close all;
%%
%������������ֱ��Ǳ���б��˻���ʹ�ô�����֧���տ��¼
filename = '201601.csv';
x=load(filename);
%x=x(1:50000,:);
name = x(:,1)';
balance = x(:,2)';
times = x(:,3)';
payAndload = x(:,4)';

%%
%������̣�Ҫ��ÿ���˽���һ�����ݣ��������ǵĵ���ʹ�ô������˻�����ֵ������֧���ܺͺ͵���������ֵ��ֵ
namebackup = unique(name);%Ԫ��ȥ��
%����ÿ���˵�ȫ��ʹ�ô���
times_out = zeros(1,length(namebackup));
for i = 1:length(namebackup)
    for j = 1:length(name)
        if namebackup(i) == name(j) 
            times_out(i) = times_out(i)+1;
        end
    end
end
%����ÿ���˵��˻�����ֵ
balance_out = zeros(1,length(namebackup));
count = 0;
num = 0;
for i = 1:length(balance_out)
    for j = 1:length(name)
        if namebackup(i) == name(j) 
            count=balance(j)+count;
            num = num+1;
        end
    end
    balance_out(i)= ceil(count/num);
    count = 0;
    num = 0;
end
%����ÿ���˵ĸ���֧���ܺͺ͵���������ֵ��ֵ
pay_out = zeros(1,length(namebackup));
invest = zeros(1,length(namebackup));
count = 0;
res = 0;
for i = 1:length(namebackup)
    for j = 1:length(name)
        if namebackup(i) == name(j) 
            if payAndload(j)<0
                count=payAndload(j)+count;
            else
                res = res + payAndload(j);
            end
        end
    end
    pay_out(i)= count;
    invest(i) = res;
    count = 0;
    res=0;
end
% ���ǩ
y = zeros(1,length(namebackup));
for i = 1:length(namebackup)
    if (floor(-pay_out(i)/times_out(i))<400 && times_out(i)>10)
        y(i) =1;
    else
        y(i)=0;
    end
end
 %%
%д��Ŀ��csv�ļ�
%��ͷ

various={'name','times','balance','allPay','invest','y',};
%�������
result_table=table(namebackup',times_out',balance_out',pay_out',invest',y','VariableNames',various);
%����csv���
writetable(result_table, 'teat_result.csv');














