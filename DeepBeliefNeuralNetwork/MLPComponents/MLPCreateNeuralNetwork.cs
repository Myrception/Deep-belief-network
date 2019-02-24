namespace DeepBeliefNeuralNetwork.MLPComponents
{
    public class MLPCreateNeuralNetwork
    {
        private int _Neurons;

        public int Neurons
        {
            get { return _Neurons; }
            set { _Neurons = value; }
        }

        private IFunktionen _ActivationFunction;

        public IFunktionen ActivationFunction
        {
            get { return _ActivationFunction; }
            set { _ActivationFunction = value; }
        }

        private IFunktionen _OutputFunction;

        public IFunktionen OutputFunction
        {
            get { return _OutputFunction; }
            set { _OutputFunction = value; }
        }
    }
}