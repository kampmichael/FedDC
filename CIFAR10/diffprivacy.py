import numpy as np

def makeAllDifferentiallyPrivate(params, old_params, dp_sigma, dp_S):
    new_params = []
    for i in range(len(params)):
        #clipping
        update = old_params[i].getCopy()
        update.scalarMultiply(-1.0)
        update.add(params[i]) #update = params[i] - pld_params[i]
        normFact = max(1, (np.linalg.norm(update.toVector())/dp_S))
        update.scalarMultiply(1./normFact) #clipp the update
        #add noise
        clipped_update = update.toVector()
        clipped_update += np.random.normal(loc=0.0, scale=(dp_sigma**2)*(dp_S**2), size=clipped_update.shape)
        update.fromVector(clipped_update)
        update.add(old_params[i]) #now we have the actual new model parameters
        new_params.append(update)
    return new_params
    
def makeDifferentiallyPrivate(param, old_param, dp_sigma, dp_S):
    #clipping
    update = old_param.getCopy()
    update.scalarMultiply(-1.0)
    update.add(param) #update = params[i] - pld_params[i]
    normFact = max(1, (np.linalg.norm(update.toVector())/dp_S))
    update.scalarMultiply(1./normFact) #clipp the update
    #add noise
    scale = (dp_sigma**2)*(dp_S**2)
    update.addNormalNoise(0.0, scale)
    update.add(old_param) #now we have the actual new model parameters
    return update
    