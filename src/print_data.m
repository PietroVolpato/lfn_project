% Dati degli esperimenti
datasets = {'Facebook', 'Biological', 'Citation', 'Proteins', 'Spam'};
algorithms = {'Node2Vec', 'LINE', 'AW'};

% Reconstruction Error (RE)
RE = [
    0.0309, 0.0277, 0.0514; % Facebook
    0.0439, 0.0438, 0.0771; % Biological
    0.0687, 0.1057, 0.1349; % Citation
    0.0002, 0.0182, 0.0005; % Proteins
    0.0978, 0.1055, 0.2398; % Spam
];

% False Positive Rate (FP Rate)
FP_rate = [
    0.0285, 0.0094, 0.0286; % Facebook
    0.0425, 0.0214, 0.0473; % Biological
    0.0597, 0.0441, 0.0907; % Citation
    0.0002, 0.0003, 0.0005; % Proteins
    0.0803, 0.0444, 0.1496; % Spam
];

% False Negative Rate (FN Rate)
FN_rate = [
    0.0024, 0.0183, 0.0228; % Facebook
    0.0014, 0.0224, 0.0298; % Biological
    0.0089, 0.0616, 0.0442; % Citation
    0.0000, 0.0179, 0.0000; % Proteins
    0.0175, 0.0611, 0.0902; % Spam
];

% Avg_Pos e Avg_Neg
avg_pos = [
    -1.6583, 0.6217, -0.8456; % Facebook
    -2.1555, 0.4663, -1.5834; % Biological
    -1.9571, 0.4714, -1.3017; % Citation
    -0.4845, 0.7594, -0.9880; % Proteins
    -2.0229, 0.3773, -1.1361; % Spam
];

avg_neg = [
    -4.0509, -0.0029, -5.3522; % Facebook
    -4.3784, 0.0177, -4.3293; % Biological
    -3.7426, 0.0118, -3.7595; % Citation
    -5.7108, -0.0008, -5.7486; % Proteins
    -3.5367, 0.0016, -2.3799; % Spam
];

% Reconstruction Error (RE)
figure;
bar(RE, 'grouped');
title('Reconstruction Error (RE)');
xlabel('Dataset');
ylabel('Reconstruction Error (RE)');
xticks(1:length(datasets));
xticklabels(datasets);
legend(algorithms, 'Location', 'northwest');
grid on;

% Save
exportgraphics(gcf, 'RE_graph.png', 'Resolution', 300);

% FP and FN Rate
figure;
subplot(2,1,1); % FP Rate
bar(FP_rate, 'grouped');
title('False Positive Rate (FP Rate)');
xlabel('Dataset');
ylabel('FP Rate');
xticks(1:length(datasets));
xticklabels(datasets);
legend(algorithms, 'Location', 'northwest');
grid on;

subplot(2,1,2); % FN Rate
bar(FN_rate, 'grouped');
title('False Negative Rate (FN Rate)');
xlabel('Dataset');
ylabel('FN Rate');
xticks(1:length(datasets));
xticklabels(datasets);
legend(algorithms, 'Location', 'northwest');
grid on;

% Save
exportgraphics(gcf, 'FP_FN_graph.png', 'Resolution', 300);

% Avg_Pos and Avg_Neg
figure;
hold on;
bar(avg_pos, 'grouped'); % Avg_Pos
bar(avg_neg, 'grouped', 'FaceAlpha', 0.5); % Avg_Neg (trasparente)
hold off;
title('Avg\_Pos e Avg\_Neg per Dataset');
xlabel('Dataset');
ylabel('Valori');
xticks(1:length(datasets));
xticklabels(datasets);
legend([algorithms, strcat(algorithms, ' (Avg\_Neg)')], 'Location', 'northoutside');
grid on;

% Save
exportgraphics(gcf, 'Avg_Pos_Neg_graph.png', 'Resolution', 300);
%%
