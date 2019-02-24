using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace DeepBeliefNeuralNetwork
{
    public class BA_Import
    {
        public List<List<PatternToLearn>> Import(string Trainingspfad, string Testpfad)
        {
            List<List<PatternToLearn>> returnList = new List<List<PatternToLearn>>();
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
            returnList.Add(pattertolearn);
            returnList.Add(pattertotest);
            returnList.Add(bildpattertotest);
            return returnList;
        }
    }
}