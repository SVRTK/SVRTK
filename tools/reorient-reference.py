#! /usr/bin/env python

import sys
import os.path
import argparse
import mirtk
import untangle


def printpointset(in_file):
    obj = untangle.parse(in_file)
    itemlist = obj.point_set_file.point_set.time_series.point
    globalpointlist = []
    
    num=0
    for s in itemlist:
        globalpointlist.append(float(s.x.cdata))
        globalpointlist.append(float(s.y.cdata))
        globalpointlist.append(float(s.z.cdata))
        num=num+1


    outstring = " "
    outstring += str(num)
    outstring += " "
    for s in itemlist:
        
        outstring += str(-float(s.x.cdata))
        outstring += " "
        outstring += str(-float(s.y.cdata))
        outstring += " "
        outstring += str(float(s.z.cdata))
        outstring += " "

    return outstring



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ti", "--targetimage",
                        type=str,
                        help="target image file")
    parser.add_argument("-si", "--sourceimage",
                        type=str,
                        help="source image file")
    parser.add_argument("-tp", "--targetpointset",
                    type=str,
                    help="4 - 5 points from MITK with order: Left, Right, Inferior, Superior, center (optional).")
    parser.add_argument("-sp", "--sourcepointset",
                        type=str,
                        help="4 - 5 points from MITK with order: Left, Right, Inferior, Superior, center (optional).")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output image file")
    parser.add_argument("-om", "--mode",
                        type=int,
                        default=0,
                        help="output format mode")
    parser.add_argument("-do", "--dofout",
                        type=str,
                        help="output dof file")
    args = parser.parse_args()
                                            
    if (args.targetimage == None or args.sourceimage == None or args.targetpointset == None or args.sourcepointset == None or args.output == None):
        parser.print_help()
        sys.exit(1)



    if not os.path.isfile(args.targetimage):
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.sourceimage):
        parser.print_help()
        sys.exit(1)
        
    if not os.path.isfile(args.targetpointset):
        parser.print_help()
        sys.exit(1)
        
    if not os.path.isfile(args.sourcepointset):
        parser.print_help()
        sys.exit(1)

#    print("------------------------------------------------------")
#    print("------------------------------------------------------")

    print ("org target:")
    target_pointset_string = printpointset(args.targetpointset)
#    print(target_pointset_string)

    print ("org source:")
    source_pointset_string = printpointset(args.sourcepointset)
#    print(source_pointset_string)

    with open("source_pointset.txt", "w") as sf:
        sf.write("%s\n" % source_pointset_string)
    
    with open("target_pointset.txt", "w") as tf:
        tf.write("%s\n" % target_pointset_string)

    
#
#    source_txt_file = open("source_pointset.txt", "w")
#    source_txt_file.write("%s" % source_pointset_string)
#    source_txt_file.close()
#
#    target_txt_file = open("target_pointset.txt", "w")
#    target_txt_file.write("%s" % target_pointset_string)
#    target_txt_file.close()

#    exit(1)


#    print("------------------------------------------------------")
#    print("------------------------------------------------------")



    mirtk.register_points(args.targetimage, args.sourceimage, args.output, args.dofout, args.mode, "target_pointset.txt", "source_pointset.txt")


#    print("------------------------------------------------------")
#    print("------------------------------------------------------")


if __name__ == "__main__":
    main()
