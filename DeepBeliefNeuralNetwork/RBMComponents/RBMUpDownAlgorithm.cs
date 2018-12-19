using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    public class RBMUpDownAlgorithm
    {
        private Random _rnd = new Random();
        private List<RBMNeurons> _initialBinaryHiddenList;
        //private List<RBMNeurons> _initialBinaryVisualList;
        private List<RBMNeurons> _reconstructedBinaryVisualList;
        public int CalculateHiddenBinaryState(double bias, RBMNeurons actualHiddeNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> visualLayer )
        {
            double sum = 0;
            for (int i = 0; i < weightMatrix.Row; i++)
            {
                sum += visualLayer[i].BinaryState*weightMatrix[i,actualHiddeNeuron.Index];
            }
            double probability = 1d / (1d + Math.Exp(-(bias + sum)));
            actualHiddeNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public int CalculateReconstructionVisualBinaryState(double bias, RBMNeurons actualVisualNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> hiddenLayer)
        {
            _initialBinaryHiddenList = hiddenLayer;
            double sum = 0;
            for (int i = 0; i < weightMatrix.Coloumn - weightMatrix.Row; i++)
            {
                sum += hiddenLayer[i].BinaryState * weightMatrix[actualVisualNeuron.Index,i];
            }
            double probability  = 1d / (1d + Math.Exp(-(bias + sum)));
            actualVisualNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public void CalculateWeightMatrixAlteration(double learningRate ,RBMWeightMatrix weightMatrix ,List<RBMNeurons> visualLayer, List<RBMNeurons> hiddenLayer, RBMBiasNeuron biasNeuron, double[]initalLearningVector)
        {
            double vectorLenght = initalLearningVector.Sum(componentOfVector => Math.Pow(componentOfVector, 2));
            vectorLenght = Math.Sqrt(vectorLenght);
            _reconstructedBinaryVisualList = visualLayer;
            for (int i = 0; i < weightMatrix.Row; i++) // Update weightmatrix
            {
                for (int j = 0; j < weightMatrix.Coloumn-weightMatrix.Row; j++)
                {
                    weightMatrix[i, j + weightMatrix.Row] += (learningRate *
                                                              ((initalLearningVector[i] *
                                                                _initialBinaryHiddenList[j].Probability) -
                                                               (_reconstructedBinaryVisualList[i].BinaryState *
                                                                hiddenLayer[j].Probability))) /
                                                             vectorLenght;
                }
            }
            for (int i = 0; i < hiddenLayer.Count; i++) // Update Bias
            {
                biasNeuron.BiasHidden += (learningRate *
                                          (_initialBinaryHiddenList[i].BinaryState - hiddenLayer[i].Probability)) /
                                         vectorLenght;
                biasNeuron.BiasVisible += (learningRate *
                                          (_initialBinaryHiddenList[i].BinaryState - hiddenLayer[i].Probability)) /
                                         vectorLenght;
            }
        }
    }
}
