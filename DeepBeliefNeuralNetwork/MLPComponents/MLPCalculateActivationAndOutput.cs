namespace DeepBeliefNeuralNetwork.MLPComponents
{
    internal static class MLPCalculateActivationAndOutput
    {
        internal static void Calculate(this MLPNeuron temp, double alpha)
        {
            temp.activation = temp.ActivationFunction.BerechneWert(temp.NetInput, alpha);
            temp.Output = temp.OutputFunction.BerechneWert(temp.activation, alpha);
        }
    }
}