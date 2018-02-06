% Dataset from UCI Repo
% http://archive.ics.uci.edu/ml/datasets/Arrhythmia
load arrhythmia;

%% Data Understanding
% See dataset description
summary(dataset(X))

% We can see that there are some missing values in column (attribute)
% 11-14, as well as their proportion

% Check position of missing value
% missing = ismissing(X); 


%% Data Pre-Processing
% Remove attr with more than 1/3 value missing
% Attr 14
X(:,14)=[];

% Impute missing value with nearest non-missing value. Other type of
% imputation is also available, e.g. previous non-missing value, next
% non-missing value, etc.
X=fillmissing(X,'nearest');

% The label of this dataset is originally multiclass. For simplicity
% reason, this code converted multiclass problem into binary problem. 
% So basically, anything beside healthy is considered sick (having arrhythmia)
Y(Y>1)=2;

% Activate this to see if the two class is balanced
% sum(Y==1)
% sum(Y==2)

% Remove attributes which has zero variance (which means that there is no
% variation among values in that attribute, or has same value for all
% observations). This kind of attributes typically is not useful in
% prediction.
nAllFeat=size(X);
nAllFeat=nAllFeat(:,2)
varAttr=[];
for k=1:nAllFeat %for all attributes, calculate variance
    varInd=var(X(:,k));
    varAttr=[varAttr varInd];
end
zeroVarAttr = find(varAttr==0); %find index of attributes with zero variance
X2= X;
X2(:,zeroVarAttr)=[]; %remove attributes with zero variance

%% Feature Selection
% There are a lot of feature selection method available. This example uses
% ranker method (e.g. RelieF, wilcoxon/Mann-Whitney test) which evaluates
% relevance of individual attributes to target class.

%f = rankfeatures(X',Y,'Criterion','wilcoxon','NumberOfIndices',25);
ff = relieff(X2,Y,10); % This will give you list of ranked feature
f = ff(:,1:150)'; %pick top 70

%% Experiment - Baseline
% Split Training-Test set with Original data
holdoutCV = cvpartition(Y,'holdout',0.2);
dataTrain = X(holdoutCV.training,:);
grpTrain = Y(holdoutCV.training,:);
dataTest = X(holdoutCV.test,:);
grpTest = Y(holdoutCV.test,:);

% Set random generator to known state.
rng('default');

% Initial Model with all features included (unranked)
mdl0 = fitctree(dataTrain,grpTrain);
pred0 = predict(mdl0,dataTrain);
%perf0 = classperf(grpTrain,pred0); %this require Bioinformatics Toolbox
%get(perf0);
AccManual0 = mean(pred0 == grpTrain) %Accuracy

% Validation
pred0t = predict(mdl0,dataTest);
%perf0t = classperf(grpTest,pred0t);
%get(perf0t);
AccManual0t = mean(pred0t == grpTest)

% Note: With low Accuracy in Test, we can see that the model is pretty 
% overfitting

%% Model with selected features
% Split Training-Test set with Original data
holdoutCV = cvpartition(Y,'holdout',0.2);
dataTrain = X2(holdoutCV.training,:);
grpTrain = Y(holdoutCV.training,:);
dataTest = X2(holdoutCV.test,:);
grpTest = Y(holdoutCV.test,:);

% Set random generator to known state.
rng('default');

% Take top f feature
mdl1 = fitctree(dataTrain(:,f),grpTrain);
pred1 = predict(mdl1,dataTrain(:,f));
AccManual1 = mean(pred1 == grpTrain)
% Validation
pred1t = predict(mdl1,dataTest(:,f));
perf1t = classperf(grpTest,pred1t);
AccManual1t = mean(pred1t == grpTest)

% Note: Test accuracy increased!

%% Experiment - Feature Selection
% Now let's see what's going on if we decided to take top 10%, 20%, 30%,
% ...to 100% of ranked features
fperc = round(linspace(1,length(ff),10),0);
%length(fperc); %just for checking
AccTrain=[];
AccTest=[];

for i=1:length(fperc)
    numFeat = fperc(:,i);
    fx=ff(:,1:numFeat);
    % Model with selected features
    mdl2 = fitctree(dataTrain(:,fx),grpTrain);
    pred2 = predict(mdl2,dataTrain(:,fx));
    perf2 = classperf(grpTrain,pred2);
    AccManual2 = mean(pred2 == grpTrain);
    AccTrain = [AccTrain AccManual2];
    % Validation
    pred2t = predict(mdl2,dataTest(:,fx));
    perf2t = classperf(grpTest,pred2t);
    AccManual2t = mean(pred2t == grpTest);
    AccTest = [AccTest AccManual2t];
end

figure
plot(fperc,AccTrain,fperc,AccTest,'--')
xlabel('# Features')
ylabel('Accuracy')
