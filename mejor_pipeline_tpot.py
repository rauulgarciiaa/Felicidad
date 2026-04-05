# Mejor pipeline encontrado por TPOT AutoML
# Accuracy en test: 0.7500

from sklearn.pipeline import make_pipeline
import joblib

# Pipeline:
# Pipeline(steps=[('robustscaler',
                 RobustScaler(quantile_range=(0.2475690452454,
                                              0.9671736154307))),
                ('selectfwe', SelectFwe(alpha=0.0012903345738)),
                ('featureunion-1',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('featureunion-2',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('mlpclassifier',
                 MLPClassifier(activation='identity', alpha=0.0355177872236,
                               hidden_layer_sizes=[298, 298],
                               learning_rate='adaptive',
                               learning_rate_init=0.0016544677381,
                               n_iter_no_change=32, random_state=42))])

# Para reutilizar el modelo:
# import joblib
# modelo = joblib.load('mejor_pipeline_tpot.pkl')
# predicciones = modelo.predict(X_nuevos)
