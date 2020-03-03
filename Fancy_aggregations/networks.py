import numpy as np
# =============================================================================
# ~ NETWORK-BASED FUSIONS
# =============================================================================

def _jaccard_affinity(logits):
    '''

    :param logits:
    :return:
    '''
    nclasificadores, muestras, _ = logits.shape

    resultados = np.zeros((nclasificadores, nclasificadores))

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            resultados[i,j] = np.mean(logits[i, :] == logits[j, :])

    return resultados

def _jaccard_positive_affinity(logits, y):
    '''

    :param logits:
    :return:
    '''
    nclasificadores, muestras, _ = logits.shape

    resultados = np.zeros((nclasificadores, nclasificadores))

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            aciertos_comunes = np.mean(logits[i, :] == logits[j, :] == y)
            aciertos_totales = logits[i, :] == y
            resultados[i,j] = aciertos_comunes / aciertos_totales

    return resultados

def lippman_decisor(X, keepdims=False, axis=0, tnorm=min, agg_function=np.max, aff_func=_jaccard_affinity):
    '''
    WARNING: Very basic implementation. No vectorization or anything.
    ONLY BINARY CLASSIFICATION
    :param X: logits, in the shape classifiers x samples x classes
    :return:
    '''
    if axis != 0:
        X = np.swapaxes(X, 0, axis)

    clasificadores, muestras, clases = X.shape
    simple_votes = np.argmax(X, axis=2) #Returns a matrix: classifier x samples (x predicted class, singleton dimension omitted)

    simple_punctuations = X[:,:,0]

    affinities = aff_func(X)

    if keepdims:
        final_votes = np.zeros((1, muestras, clases))
    else:
        final_votes = np.zeros(( muestras, clases))

    for muestra in range(muestras):
        scores = [0] * clases

        #Phase 1:obtain the 'original thinking' reward.
        for c in range(clases):
            for i in range(clasificadores):
                for j in range(clasificadores):
                    if i != j:
                        if (simple_votes[i, muestra]  == c) and (simple_votes[j, muestra] == c):
                            scores[c] += tnorm((1-affinities[i,j]), simple_punctuations[i, muestra])
                        elif (simple_votes[i, muestra] == c) and (simple_votes[j, muestra] != c):
                            scores[c] += tnorm(affinities[i,j], simple_punctuations[i, muestra])

        #Phase 2: Obtain the 'basic' votation.
        m_simple_votes = simple_punctuations[:, muestra]

        scores[0] = scores[0] / clasificadores
        scores[1] = scores[1] / clasificadores

        scores[0] += agg_function(m_simple_votes)
        scores[1] += agg_function(1 - m_simple_votes)

        if keepdims:
            final_votes[0, muestra, 0] = scores[0]
            final_votes[0, muestra, 1] = scores[1]
        else:
            final_votes[muestra,0] = scores[0]
            final_votes[muestra,1] = scores[1]

    if axis != 0:
        X = np.swapaxes(X, axis, 0)
    return final_votes

def lucrezia_simple_decisor(X, keepdims=False, axis=0, tnorm=min, agg_function=np.max, aff_func=_jaccard_affinity, labels=None):
    '''
    WARNING: Very basic implementation. No vectorization or anything.
    ONLY BINARY CLASSIFICATION
    :param X: logits, in the shape classifiers x samples x classes
    :return:
    '''
    if axis != 0:
        X = np.swapaxes(X, 0, axis)

    clasificadores, muestras, clases = X.shape
    simple_votes = np.argmax(X, axis=2) #Returns a matrix: classifier x samples (x predicted class, singleton dimension omitted)
    simple_punctuations = X[:, :, 0]

    if not (labels is None):
        autority = np.zeros((clasificadores,))
        for clasificador in range(clasificadores):
            autority[clasificador] = np.mean(np.equal(labels, simple_votes[clasificador, :]))

    else:
        autority = None

    affinities = aff_func(X)

    if keepdims:
        final_votes = np.zeros((1, muestras, clases))
    else:
        final_votes = np.zeros((muestras, clases))

    for muestra in range(muestras):
        scores = [0] * clases

        for i in range(clasificadores):
            common_prediction_classifiers = simple_votes[:, muestra] == simple_votes[i, muestra]
            enemy_prediction_classifiers = simple_votes[:, muestra] != simple_votes[i, muestra]
            try:
                consensus = agg_function(affinities[i, common_prediction_classifiers])
            except ValueError:
                consensus = 0
            try:
                disension = agg_function(affinities[i, enemy_prediction_classifiers])
            except ValueError:
                disension = 0

            if not (autority is None):
                    scores[0] += tnorm(autority[i], tnorm(simple_punctuations[i, muestra], consensus))
                    scores[1] += tnorm(autority[i], tnorm((1 - simple_punctuations[i, muestra]), consensus))
            else:
                    scores[0] += tnorm(simple_punctuations[i, muestra], consensus)
                    scores[1] += tnorm((1 - simple_punctuations[i, muestra]), consensus)

        if keepdims:
            final_votes[0, muestra, 0] = scores[0]
            final_votes[0, muestra, 1] = scores[1]
        else:
            final_votes[muestra, 0] = scores[0]
            final_votes[muestra, 1] = scores[1]

    if axis != 0:
        X = np.swapaxes(X, axis, 0)

    return final_votes