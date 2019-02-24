namespace DeepBeliefNeuralNetwork.MLPComponents
{
    public class MLPNeuron
    {
        private int _Index;

        public int Index
        {
            get { return _Index; }
            set { _Index = value; }
        }

        private double _NettoInput;

        public double NetInput
        {
            get { return _NettoInput; }
            set { _NettoInput = value; }
        }

        private double _activation;

        public double activation
        {
            get { return _activation; }
            set { _activation = value; }
        }

        private double _Output;

        public double Output
        {
            get { return _Output; }
            set { _Output = value; }
        }

        public IFunktionen ActivationFunction { get; set; }
        public IFunktionen OutputFunction { get; set; }
    }
}