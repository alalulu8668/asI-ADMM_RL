function featurize = phi_s(state,syspar)

switch syspar.mapping 
    case 'rbf'
    rbf = syspar.rbf;
    % normalization
    weight = rbf.weight;
    offset = rbf.offset;
    n_component = rbf.n_component;

    state = (state-rbf.x_mean)./rbf.x_std;

    % featurize = [];

    projection = weight'*state + offset';
    projection = sqrt(2/n_component).*cos(projection);
    featurize = projection;

    case 'table'
        numObs = syspar.numObs;
        featurize = ind2vec(state,numObs);
    case 'table_1'
        numObs = syspar.numObs;
        if state == 0
            featurize = ind2vec(numObs,numObs);
        else
            featurize = ind2vec(state,numObs);
        end
end