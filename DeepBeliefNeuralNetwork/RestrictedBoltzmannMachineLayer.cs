using DeepBeliefNeuralNetwork.RBMComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
    public class RestrictedBoltzmannMachineLayer
    {
        public List<RBMNeurons> VisualNeurons = new List<RBMNeurons>();
        public List<RBMNeurons> HiddenNeurons = new List<RBMNeurons>();
        public RBMBiasNeuron BiasNeuron = new RBMBiasNeuron();
        public double[] InitialLearningVector;
        public RBMWeightMatrix WeightMatrix { get; set; }
        private double _learningRate;
        public RestrictedBoltzmannMachineLayer(int numberOfVisualNeurons, int numberOfHiddenNeurons, double learningRate)
        {
            int index = 0;
            WeightMatrix = new RBMWeightMatrix(numberOfVisualNeurons, numberOfVisualNeurons + numberOfHiddenNeurons);
            for (int i = 0; i < numberOfVisualNeurons; i++)
            {
                VisualNeurons.Add(new RBMNeurons() { Index = index});
                index++;
            }
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                HiddenNeurons.Add(new RBMNeurons() { Index = index });
                index++;
            }
            _learningRate = learningRate;
            WeightMatrixRandom(numberOfVisualNeurons, numberOfHiddenNeurons);
        }

        public void UpDownAlgorithm(double[] trainingVector, double errorTolerance, int contrastiveDivergenceN)
        {
            InitialLearningVector = new double[trainingVector.Length];
            InitialLearningVector = trainingVector;
            RBMUpDownAlgorithm training = new RBMUpDownAlgorithm();
            double absError, squaredError;
            for (int i = 0; i < trainingVector.Length; i++)
            {
                VisualNeurons[i].BinaryState = trainingVector[i];
            }
            do
            {
                absError = 0;
                squaredError = 0;
                    foreach (var hiddenNeuron in HiddenNeurons) // Propagate Up
                    {
                        hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(BiasNeuron.BiasHidden, hiddenNeuron, WeightMatrix, VisualNeurons);
                    }
                for (int i = 0; i < contrastiveDivergenceN; i++)
                {
                    foreach (var visualNeuron in VisualNeurons) // Propagate Down, Sample Visible (negative)
                    {
                        visualNeuron.BinaryState = training.CalculateReconstructionVisualBinaryState(BiasNeuron.BiasVisible, visualNeuron, WeightMatrix, HiddenNeurons);
                    }
                    foreach (var hiddenNeuron in HiddenNeurons) // Propagate Up, Sample Hidden (negative)
                    {
                        hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(BiasNeuron.BiasHidden, hiddenNeuron, WeightMatrix, VisualNeurons);
                    } 
                }
                training.CalculateWeightMatrixAlteration(_learningRate, WeightMatrix, VisualNeurons,HiddenNeurons, BiasNeuron,
                    InitialLearningVector);
                for (int i = 0; i < trainingVector.Length; i++)
                {
                    absError += Math.Abs(trainingVector[i]-VisualNeurons[i].BinaryState);//Absolute Fehler
                    squaredError += Math.Pow(trainingVector[i] - VisualNeurons[i].BinaryState, 2);
                }
                Console.WriteLine("ABSError:"+absError+" "+"squaredError:"+squaredError);
            } while (squaredError > errorTolerance);
        }

        //public double[] CalculateOutput(double[] inputvector)
        //{
            
        //}
        private void WeightMatrixRandom(int numberOfVisualNeurons, int numberOfHiddenNeurons)
        {
            Random RND = new Random();
            for (int i = numberOfVisualNeurons; i < numberOfVisualNeurons+numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfVisualNeurons; j++)
                {
                    WeightMatrix[j, i] = RND.NextDouble();
                }
            }
        } 
    }
}
