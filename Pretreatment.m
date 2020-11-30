
close all;
%%
%定义初试量，分别是编号列表，账户余额，使用次数和支付收款记录
filename = '201601.csv';
x=load(filename);
%x=x(1:50000,:);
name = x(:,1)';
balance = x(:,2)';
times = x(:,3)';
payAndload = x(:,4)';

%%
%计算过程：要给每个人建立一行数据，包含他们的当月使用次数，账户余额均值，负数支出总和和单笔正数充值均值
namebackup = unique(name);%元素去重
%创建每个人的全月使用次数
times_out = zeros(1,length(namebackup));
for i = 1:length(namebackup)
    for j = 1:length(name)
        if namebackup(i) == name(j) 
            times_out(i) = times_out(i)+1;
        end
    end
end
%创建每个人的账户余额均值
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
%创建每个人的负数支出总和和单笔正数充值均值
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
% 打标签
y = zeros(1,length(namebackup));
for i = 1:length(namebackup)
    if (floor(-pay_out(i)/times_out(i))<400 && times_out(i)>10)
        y(i) =1;
    else
        y(i)=0;
    end
end
 %%
%写入目标csv文件
%表头

various={'name','times','balance','allPay','invest','y',};
%表的内容
result_table=table(namebackup',times_out',balance_out',pay_out',invest',y','VariableNames',various);
%创建csv表格
writetable(result_table, 'teat_result.csv');














