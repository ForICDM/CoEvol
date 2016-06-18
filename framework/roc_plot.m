function AUC = roc_plot(labels, scores)

positive_class = 1;
[X,Y,~,AUC] = perfcurve(labels, scores, positive_class);
plot(X,Y)
xlabel('False Positive Rate'); 
ylabel('True Positive Rate')
hold on
plot([0 1],[0 1],'--')

% hold on
% [~, I] = sort(scores);
% labels = flip(labels(I));
% for i = 1:length(scores)
%     tpr(i) = length(find(labels(1:i)==1))/length(find(labels==1));
%     fpr(i) = length(find(labels(1:i)==0))/length(find(labels==0));
% end
% plot(fpr,tpr, '--')
% return

% hold on
% % Obtain errors on TPR by vertical averaging
% [X,Y] = perfcurve(labels, scores, positive_class,'nboot',100,'xvals','all');
% errorbar(X,Y(:,1),Y(:,1)-Y(:,2),Y(:,3)-Y(:,1)); % plot errors

end