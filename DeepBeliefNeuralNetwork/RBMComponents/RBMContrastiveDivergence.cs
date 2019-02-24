using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    internal class RBMContrastiveDivergence
    {
        //private Random _rnd = new Random();
        //private List<RBMNeurons> _initialBinaryHiddenList;
        //private List<RBMNeurons> _initialBinaryVisualList;
        private List<RBMNeurons> _reconstructedBinaryVisualList;

        private int a = 1;

        public int CalculateHiddenBinaryState(double bias, RBMNeurons actualHiddeNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> visualLayer, int weightMatrixRow, ThreadSafeRandom _rnd)
        {
            double sum = 0;
            for (int i = 0; i < weightMatrixRow; i++)
            {
                sum += visualLayer[i].BinaryState * weightMatrix[i, actualHiddeNeuron.Index];
            }
            double probability = 1d / (1d + Math.Exp(-(bias + sum / a)));
            actualHiddeNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public int CalculateHiddenBinaryStateWithProbability(double bias, RBMNeurons actualHiddeNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> visualLayer, int weightMatrixRow, ThreadSafeRandom _rnd)
        {
            double sum = 0;
            for (int i = 0; i < weightMatrixRow; i++)
            {
                sum += visualLayer[i].Probability * weightMatrix[i, actualHiddeNeuron.Index];
            }
            double probability = 1d / (1d + Math.Exp(-(bias + sum / a)));
            actualHiddeNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public int CalculateReconstructionVisualBinaryStateDrivenByData(double bias, RBMNeurons actualVisualNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> hiddenLayer, int weightMatrixRow, int weightMatrixColoumn, ThreadSafeRandom _rnd)
        {
            //_initialBinaryHiddenList = hiddenLayer;
            double sum = 0;
            for (int i = weightMatrixRow; i < weightMatrixRow + weightMatrixColoumn; i++)
            {
                sum += hiddenLayer[i - weightMatrixRow].BinaryState * weightMatrix[actualVisualNeuron.Index, i];
            }
            double probability = 1d / (1d + Math.Exp(-(bias + sum / a)));
            actualVisualNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public int CalculateReconstructionVisualBinaryStateDrivenByReconstruction(double bias, RBMNeurons actualVisualNeuron, RBMWeightMatrix weightMatrix, List<RBMNeurons> hiddenLayer, int weightMatrixRow, int weightMatrixColoumn, ThreadSafeRandom _rnd)
        {
            double sum = 0;
            for (int i = weightMatrixRow; i < weightMatrixRow + weightMatrixColoumn; i++)
            {
                sum += hiddenLayer[i - weightMatrixRow].Probability * weightMatrix[actualVisualNeuron.Index, i];
            }
            double probability = 1d / (1d + Math.Exp(-(bias + sum / a)));
            actualVisualNeuron.Probability = probability;
            return _rnd.NextDouble() <= probability ? 1 : 0;
        }

        public void CalculateWeightMatrixAlteration(double learningRate, RBMWeightMatrix weightMatrix, List<RBMNeurons> visualLayer, List<RBMNeurons> hiddenLayer, RBMBiasNeuron biasNeuron, double[] initalLearningVector, double[] _initialBinaryHiddenList, int WeightMatrixRow, int WeightMatrixColoumn)
        {
            double vectorLenght = initalLearningVector.Sum(componentOfVector => Math.Pow(componentOfVector, 2));
            //vectorLenght = vectorLenght > 0 || vectorLenght < 0 ? Math.Sqrt(vectorLenght) : 1d;
            vectorLenght = 1d;
            _reconstructedBinaryVisualList = visualLayer;
            for (int i = 0; i < WeightMatrixRow; i++) // Update weightmatrix
            {
                for (int j = 0; j < WeightMatrixColoumn - WeightMatrixRow; j++)
                {
                    weightMatrix[i, j + WeightMatrixRow] += (learningRate *
                                                              ((initalLearningVector[i] *
                                                                _initialBinaryHiddenList[j]) -
                                                               (_reconstructedBinaryVisualList[i].BinaryState *
                                                                hiddenLayer[j].Probability))) /
                                                             vectorLenght;
                    weightMatrix[j + WeightMatrixRow, i] = weightMatrix[i, j + WeightMatrixRow];
                }
            }
            for (int i = 0; i < initalLearningVector.Length; i++) // Update Bias
            {
                biasNeuron.BiasVisible += (learningRate *
                                          (initalLearningVector[i] - _reconstructedBinaryVisualList[i].Probability)) /
                                         initalLearningVector.Length;
            }
            for (int i = 0; i < hiddenLayer.Count; i++)
            {
                biasNeuron.BiasHidden += (learningRate *
                                         (_initialBinaryHiddenList[i] - hiddenLayer[i].Probability)) /
                                        hiddenLayer.Count;
            }
        }
    }
}