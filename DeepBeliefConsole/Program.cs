using DeepBeliefNeuralNetwork;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DeepBeliefConsole
{
    internal class Program
    {
        /// <summary>
        /// Großteil aller einstellungen sind in der Klasse DeepBeliefNetwork zu finden.
        /// </summary>
        /// <param name="args"></param>
        private static void Main(string[] args)
        {
            string Desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string OrdnerBachelorarbeit = "BachelorarbeitJoseph";
            string Ordnerpfad = Desktop + @"\" + OrdnerBachelorarbeit;
            string Trainingspfad = Ordnerpfad + @"\Noise\Training\Mean0Variance2";
            string Testpfad = Ordnerpfad + @"\Noise\gaussian\Mean0Variance2";

            string Variance = Testpfad.Split(@"\\".ToCharArray())[Testpfad.Split(@"\\".ToCharArray()).Length - 1];

            List<PatternToLearn> pattertolearn = new List<PatternToLearn>();
            List<PatternToLearn> pattertotest = new List<PatternToLearn>();
            List<PatternToLearn> bildpattertotest = new List<PatternToLearn>();

            int targetCounter = 0;

            var bilder = new BildLaden();
            Bitmap bild;
            foreach (var directories in Directory.EnumerateDirectories(Trainingspfad))
            {
                foreach (var file in Directory.EnumerateFiles(directories, "*b.bmp"))
                {
                    bild = bilder.LoadPicture(file);

                    PatternToLearn muster = new PatternToLearn();
                    muster.inputvector = new double[784];
                    muster.targetvector = new double[3];
                    int counter = 0;
                    for (int i = 0; i < bild.Height; i++)
                    {
                        for (int j = 0; j < bild.Width; j++)
                        {
                            Color farbe = bild.GetPixel(i, j);
                            if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                            {
                                muster.inputvector[counter] = 0;
                            }
                            else
                            {
                                muster.inputvector[counter] = 1;
                            }
                            counter++;
                        }
                    }
                    muster.targetvector[targetCounter] = 1;
                    pattertolearn.Add(muster);
                }
                foreach (var file in Directory.EnumerateFiles(directories, "*a.bmp"))
                {
                    bild = bilder.LoadPicture(file);

                    PatternToLearn muster = new PatternToLearn();
                    muster.inputvector = new double[784];
                    muster.targetvector = new double[3];
                    int counter = 0;
                    for (int i = 0; i < bild.Height; i++)
                    {
                        for (int j = 0; j < bild.Width; j++)
                        {
                            Color farbe;
                            farbe = bild.GetPixel(i, j);
                            if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                            {
                                muster.inputvector[counter] = 0;
                            }
                            else
                            {
                                muster.inputvector[counter] = 1;
                            }
                            counter++;
                        }
                    }
                    muster.targetvector[targetCounter] = 1;
                    pattertolearn.Add(muster);
                }
                targetCounter++;
            }

            //bild = bilder.LoadPicture(@"C:\Users\Joseph\Desktop\Projekte\hitz\Pictures\Spiral1\0a.bmp");

            //var amuster = new PatternToLearn
            //{
            //    inputvector = new double[784],
            //    targetvector = new double[3]
            //};
            //int acounter = 0;
            //for (int i = 0; i < bild.Height; i++)
            //{
            //    for (int j = 0; j < bild.Width; j++)
            //    {
            //        var farbe = bild.GetPixel(i, j);
            //        if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
            //        {
            //            amuster.inputvector[acounter] = 0;
            //        }
            //        else
            //        {
            //            amuster.inputvector[acounter] = 1;
            //        }
            //        acounter++;
            //    }
            //}
            //bildpattertotest.Add(amuster);

            targetCounter = 0;

            foreach (var directories in Directory.EnumerateDirectories(Testpfad))
            {
                foreach (var file in Directory.EnumerateFiles(directories, "*b.bmp"))
                {
                    bild = bilder.LoadPicture(file);

                    var muster = new PatternToLearn
                    {
                        inputvector = new double[784],
                        targetvector = new double[3]
                    };
                    int counter = 0;
                    for (int i = 0; i < bild.Height; i++)
                    {
                        for (int j = 0; j < bild.Width; j++)
                        {
                            var farbe = bild.GetPixel(i, j);
                            if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                            {
                                muster.inputvector[counter] = 0;
                            }
                            else
                            {
                                muster.inputvector[counter] = 1;
                            }
                            counter++;
                        }
                    }
                    muster.targetvector[targetCounter] = 1;
                    pattertotest.Add(muster);
                }
                foreach (var file in Directory.EnumerateFiles(directories, "*a.bmp"))
                {
                    bild = bilder.LoadPicture(file);

                    var muster = new PatternToLearn
                    {
                        inputvector = new double[784],
                        targetvector = new double[3]
                    };
                    int counter = 0;
                    for (int i = 0; i < bild.Height; i++)
                    {
                        for (int j = 0; j < bild.Width; j++)
                        {
                            var farbe = bild.GetPixel(i, j);
                            if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                            {
                                muster.inputvector[counter] = 0;
                            }
                            else
                            {
                                muster.inputvector[counter] = 1;
                            }
                            counter++;
                        }
                    }
                    muster.targetvector[targetCounter] = 1;
                    pattertotest.Add(muster);
                }
                targetCounter++;
            }

            #region csv Import

            //string fileName = @"C:\Users\Joseph\Desktop\TrainingsMuster50.csv"; // Muss angepasst werden
            ////string TestfileName = @"C:\Users\Joseph\Desktop\TestMuster50.csv"; // Muss angepasst werden
            ////string bildTestfileName = @"C:\Users\Joseph\Desktop\0a.csv"; // Muss angepasst werden
            //////string fileName = @"C:\Users\Joseph\Desktop\mnist50.csv"; // Muss angepasst werden
            //////string TestfileName = @"C:\Users\Joseph\Desktop\mnist50.csv"; // Muss angepasst werden
            //////string bildTestfileName = @"C:\Users\Joseph\Desktop\mnist1.csv"; // Muss angepasst werden
            //StreamReader reader = new StreamReader(fileName);
            //string line;
            //string[] splitted;
            //char[] separator = ";".ToCharArray();
            ////char[] separator = ",".ToCharArray();
            //////List<PatternToLearn> pattertolearn = new List<PatternToLearn>();
            //////List<PatternToLearn> pattertotest = new List<PatternToLearn>();
            //////List<PatternToLearn> bildpattertotest = new List<PatternToLearn>();

            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    pattertolearn.Add(muster);
            //}
            //reader = new StreamReader(TestfileName);
            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    pattertotest.Add(muster);
            //}
            //reader = new StreamReader(bildTestfileName);
            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    bildpattertotest.Add(muster);
            //}

            #endregion csv Import

            #region XOR Import

            //List<PatternToLearn> XORpattertolearn = new List<PatternToLearn>();
            //List<PatternToLearn> XORpattertotest = new List<PatternToLearn>();
            //List<PatternToLearn> XORbildpattertotest = new List<PatternToLearn>();

            //XORpattertolearn.Add(new PatternToLearn() {inputvector = new double[2] {0d, 0d}, targetvector = new double[1] {0d}});
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 0d }, targetvector = new double[1] { 1d } });
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 0d, 1d }, targetvector = new double[1] { 1d } });
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 1d }, targetvector = new double[1] { 0d } });

            //XORpattertotest = XORpattertolearn;
            //XORbildpattertotest.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 0d }, targetvector = new double[1] { 1d } });

            #endregion XOR Import

            //foreach (var directories in Directory.EnumerateDirectories(Trainingspfad))
            //{
            //    foreach (var file in Directory.EnumerateFiles(directories, "*b.bmp"))
            //string Matrix = @"C:\Users\Joseph\Desktop\Projekte\Bachelorarbeit\Variante 1\ERS\Tanh\CD1\784_50_10_3_08.07.17_201535Matrix.csv";
            //string Matrix = args[0]; C:\Users\Joseph\Desktop\Projekte\Bachelorarbeit\Auswertung\Variante1\backprop\Tanh
            //string noiseDirectoryName = noisedirectories.Split(@"\\".ToCharArray())[noisedirectories.Split(@"\\".ToCharArray()).Length-1];
            //string matrixpfad = @"C:\Users\Joseph\Desktop\Projekte\Bachelorarbeit\Variante 2\ERS2\";
            //string speicherpfad = @"C:\Users\Joseph\Desktop\Projekte\Bachelorarbeit\Auswertung\Variante2\ERS2\" + noiseDirectoryName;
            //int matrixcounter = 0;
            //int anzahlMatrizen = Directory.EnumerateFiles(matrixpfad, "*Matrix.csv").Count();

            //foreach (var matrix in Directory.EnumerateFiles(matrixpfad, "*Matrix.csv"))
            //{
            //    string[] dateipfadsplitted = matrix.Split(@"\\".ToCharArray());
            //    string dateipfadBinary = "";
            //    for (int i = 0; i < dateipfadsplitted.Length - 1; i++)
            //    {
            //        dateipfadBinary += dateipfadsplitted[i] + @"\";
            //    }
            //    dateipfadBinary = dateipfadBinary + "Einstellungen.txt";

            //    var binary = new BinaryForNetworkToLoad(dateipfadBinary);

            //    MLPwithoutTraining auswertung = new MLPwithoutTraining(binary);
            //    auswertung.ErstellenErgebniss(matrix, pattertotest, bildpattertotest, speicherpfad);

            //    matrixcounter++;
            //    Console.WriteLine(matrixcounter + "/" + anzahlMatrizen);
            //}

            for (int i = 0; i < 5; i++)
            {
                DeepBeliefNetwork DBNN = new DeepBeliefNetwork();
                DBNN.GreedyLayerWiseTraining(pattertolearn, pattertotest, bildpattertotest, Variance);
            }

            //for (int i = 0; i < 30; i++)
            //{
            //    Console.Beep();
            //}
            Console.WriteLine("Fertig :D");
            Console.ReadKey();
        }
    }
}