##########################################################################
# By Corrado Chisari - http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
# Released under LGPL.
#
##########################################################################
#    GNU LESSER GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
#
#   This version of the GNU Lesser General Public License incorporates
# the terms and conditions of version 3 of the GNU General Public
# License, supplemented by the additional permissions listed below.
#
#   0. Additional Definitions.
#
#   As used herein, "this License" refers to version 3 of the GNU Lesser
# General Public License, and the "GNU GPL" refers to version 3 of the GNU
# General Public License.
#
#   "The Library" refers to a covered work governed by this License,
# other than an Application or a Combined Work as defined below.
#
#   An "Application" is any work that makes use of an interface provided
# by the Library, but which is not otherwise based on the Library.
# Defining a subclass of a class defined by the Library is deemed a mode
# of using an interface provided by the Library.
#
#   A "Combined Work" is a work produced by combining or linking an
# Application with the Library.  The particular version of the Library
# with which the Combined Work was made is also called the "Linked
# Version".
#
#   The "Minimal Corresponding Source" for a Combined Work means the
# Corresponding Source for the Combined Work, excluding any source code
# for portions of the Combined Work that, considered in isolation, are
# based on the Application, and not on the Linked Version.
#
#   The "Corresponding Application Code" for a Combined Work means the
# object code and/or source code for the Application, including any data
# and utility programs needed for reproducing the Combined Work from the
# Application, but excluding the System Libraries of the Combined Work.
#
#   1. Exception to Section 3 of the GNU GPL.
#
#   You may convey a covered work under sections 3 and 4 of this License
# without being bound by section 3 of the GNU GPL.
#
#   2. Conveying Modified Versions.
#
#   If you modify a copy of the Library, and, in your modifications, a
# facility refers to a function or data to be supplied by an Application
# that uses the facility (other than as an argument passed when the
# facility is invoked), then you may convey a copy of the modified
# version:
#
#    a) under this License, provided that you make a good faith effort to
#    ensure that, in the event an Application does not supply the
#    function or data, the facility still operates, and performs
#    whatever part of its purpose remains meaningful, or
#
#    b) under the GNU GPL, with none of the additional permissions of
#    this License applicable to that copy.
#
#   3. Object Code Incorporating Material from Library Header Files.
#
#   The object code form of an Application may incorporate material from
# a header file that is part of the Library.  You may convey such object
# code under terms of your choice, provided that, if the incorporated
# material is not limited to numerical parameters, data structure
# layouts and accessors, or small macros, inline functions and templates
# (ten or fewer lines in length), you do both of the following:
#
#    a) Give prominent notice with each copy of the object code that the
#    Library is used in it and that the Library and its use are
#    covered by this License.
#
#    b) Accompany the object code with a copy of the GNU GPL and this license
#    document.
#
#   4. Combined Works.
#
#   You may convey a Combined Work under terms of your choice that,
# taken together, effectively do not restrict modification of the
# portions of the Library contained in the Combined Work and reverse
# engineering for debugging such modifications, if you also do each of
# the following:
#
#    a) Give prominent notice with each copy of the Combined Work that
#    the Library is used in it and that the Library and its use are
#    covered by this License.
#
#    b) Accompany the Combined Work with a copy of the GNU GPL and this license
#    document.
#
#    c) For a Combined Work that displays copyright notices during
#    execution, include the copyright notice for the Library among
#    these notices, as well as a reference directing the user to the
#    copies of the GNU GPL and this license document.
#
#    d) Do one of the following:
#
#        0) Convey the Minimal Corresponding Source under the terms of this
#        License, and the Corresponding Application Code in a form
#        suitable for, and under terms that permit, the user to
#        recombine or relink the Application with a modified version of
#        the Linked Version to produce a modified Combined Work, in the
#        manner specified by section 6 of the GNU GPL for conveying
#        Corresponding Source.
#
#        1) Use a suitable shared library mechanism for linking with the
#        Library.  A suitable mechanism is one that (a) uses at run time
#        a copy of the Library already present on the user's computer
#        system, and (b) will operate properly with a modified version
#        of the Library that is interface-compatible with the Linked
#        Version.
#
#    e) Provide Installation Information, but only if you would otherwise
#    be required to provide such information under section 6 of the
#    GNU GPL, and only to the extent that such information is
#    necessary to install and execute a modified version of the
#    Combined Work produced by recombining or relinking the
#    Application with a modified version of the Linked Version. (If
#    you use option 4d0, the Installation Information must accompany
#    the Minimal Corresponding Source and Corresponding Application
#    Code. If you use option 4d1, you must provide the Installation
#    Information in the manner specified by section 6 of the GNU GPL
#    for conveying Corresponding Source.)
#
#   5. Combined Libraries.
#
#   You may place library facilities that are a work based on the
# Library side by side in a single library together with other library
# facilities that are not Applications and are not covered by this
# License, and convey such a combined library under terms of your
# choice, if you do both of the following:
#
#    a) Accompany the combined library with a copy of the same work based
#    on the Library, uncombined with any other library facilities,
#    conveyed under the terms of this License.
#
#    b) Give prominent notice with the combined library that part of it
#    is a work based on the Library, and explaining where to find the
#    accompanying uncombined form of the same work.
#
#   6. Revised Versions of the GNU Lesser General Public License.
#
#   The Free Software Foundation may publish revised and/or new versions
# of the GNU Lesser General Public License from time to time. Such new
# versions will be similar in spirit to the present version, but may
# differ in detail to address new problems or concerns.
#
#   Each version is given a distinguishing version number. If the
# Library as you received it specifies that a certain numbered version
# of the GNU Lesser General Public License "or any later version"
# applies to it, you have the option of following the terms and
# conditions either of that published version or of any later version
# published by the Free Software Foundation. If the Library as you
# received it does not specify a version number of the GNU Lesser
# General Public License, you may choose any version of the GNU Lesser
# General Public License ever published by the Free Software Foundation.
#
#   If the Library as you received it specifies that a proxy can decide
# whether future versions of the GNU Lesser General Public License shall
# apply, that proxy's public statement of acceptance of any version is
# permanent authorization for you to choose that version for the
# Library.
##########################################################################
import math
from numpy import *


def i4_bit_hi1(n):
    """Returns the position of the highest 1 bit in the 
	base 2 expansion of the integer n.
   Example:
        N    Binary     BIT
     ----    --------  ----
        0           0     0
        1           1     1
        2          10     2
        3          11     2 
        8        1000     4
       17       10001     5
     1023  1111111111    10
     """
    i = math.floor(n)
    bit = 0
    while 1:
        if i <= 0:
            break
        bit += 1
        i = math.floor(i / 2.)
    return bit


def i4_bit_lo0(n):
    """Returns the position of the low 0 bit base 2 in an integer.
    Example:
 
        N    Binary     BIT
     ----    --------  ----
        0           0     1
        1           1     2
        2          10     1
        3          11     3 
        4         100     1
        5         101     2
        6         110     1
	"""
    bit = 0
    i = math.floor(n)
    while 1:
        bit = bit + 1
        i2 = math.floor(i / 2.)
        if i == 2 * i2:
            break

        i = i2
    return bit


def i4_sobol(dim_num, seed):
    """Generates a new quasirandom Sobol vector with each call."""
    # 	Parameters:
    #
    # 		Input, integer DIM_NUM, the number of spatial dimensions.
    # 		DIM_NUM must satisfy 1 <= DIM_NUM <= 40.
    #
    # 		Input/output, integer SEED, the "seed" for the sequence.
    # 		This is essentially the index in the sequence of the quasirandom
    # 		value to be generated.	On output, SEED has been set to the
    # 		appropriate next value, usually simply SEED+1.
    # 		If SEED is less than 0 on input, it is treated as though it were 0.
    # 		An input value of 0 requests the first (0-th) element of the sequence.
    #
    # 		Output, real QUASI(DIM_NUM), the next quasirandom vector.
    #
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if not 'initialized' in list(globals().keys()):
        initialized = 0
        dim_num_save = -1

    if not initialized or dim_num != dim_num_save:
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1
        #
        # 	Initialize (part of) V.
        #
        v = zeros((dim_max, log_max))
        v[0:40, 0] = transpose(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )

        v[2:40, 1] = transpose(
            [
                1,
                3,
                1,
                3,
                1,
                3,
                3,
                1,
                3,
                1,
                3,
                1,
                3,
                1,
                1,
                3,
                1,
                3,
                1,
                3,
                1,
                3,
                3,
                1,
                3,
                1,
                3,
                1,
                3,
                1,
                1,
                3,
                1,
                3,
                1,
                3,
                1,
                3,
            ]
        )

        v[3:40, 2] = transpose(
            [
                7,
                5,
                1,
                3,
                3,
                7,
                5,
                5,
                7,
                7,
                1,
                3,
                3,
                7,
                5,
                1,
                1,
                5,
                3,
                3,
                1,
                7,
                5,
                1,
                3,
                3,
                7,
                5,
                1,
                1,
                5,
                7,
                7,
                5,
                1,
                3,
                3,
            ]
        )

        v[5:40, 3] = transpose(
            [
                1,
                7,
                9,
                13,
                11,
                1,
                3,
                7,
                9,
                5,
                13,
                13,
                11,
                3,
                15,
                5,
                3,
                15,
                7,
                9,
                13,
                9,
                1,
                11,
                7,
                5,
                15,
                1,
                15,
                11,
                5,
                3,
                1,
                7,
                9,
            ]
        )

        v[7:40, 4] = transpose(
            [
                9,
                3,
                27,
                15,
                29,
                21,
                23,
                19,
                11,
                25,
                7,
                13,
                17,
                1,
                25,
                29,
                3,
                31,
                11,
                5,
                23,
                27,
                19,
                21,
                5,
                1,
                17,
                13,
                7,
                15,
                9,
                31,
                9,
            ]
        )

        v[13:40, 5] = transpose(
            [
                37,
                33,
                7,
                5,
                11,
                39,
                63,
                27,
                17,
                15,
                23,
                29,
                3,
                21,
                13,
                31,
                25,
                9,
                49,
                33,
                19,
                29,
                11,
                19,
                27,
                15,
                25,
            ]
        )

        v[19:40, 6] = transpose(
            [
                13,
                33,
                115,
                41,
                79,
                17,
                29,
                119,
                75,
                73,
                105,
                7,
                59,
                65,
                21,
                3,
                113,
                61,
                89,
                45,
                107,
            ]
        )

        v[37:40, 7] = transpose([7, 23, 39])
        #
        # 	Set POLY.
        #
        poly = [
            1,
            3,
            7,
            11,
            13,
            19,
            25,
            37,
            59,
            47,
            61,
            55,
            41,
            67,
            97,
            91,
            109,
            103,
            115,
            131,
            193,
            137,
            145,
            143,
            241,
            157,
            185,
            167,
            229,
            171,
            213,
            191,
            253,
            203,
            211,
            239,
            247,
            285,
            369,
            299,
        ]

        atmost = 2 ** log_max - 1
        #
        # 	Find the number of bits in ATMOST.
        #
        maxcol = i4_bit_hi1(atmost)
        #
        # 	Initialize row 1 of V.
        #
        v[0, 0:maxcol] = 1

    #
    # 	Things to do only if the dimension changed.
    #
    if dim_num != dim_num_save:
        #
        # 	Check parameters.
        #
        if dim_num < 1 or dim_max < dim_num:
            print('I4_SOBOL - Fatal error!')
            print('	The spatial dimension DIM_NUM should satisfy:')
            print('		1 <= DIM_NUM <= %d' % dim_max)
            print('	But this input value is DIM_NUM = %d' % dim_num)
            return

        dim_num_save = dim_num
        #
        # 	Initialize the remaining rows of V.
        #
        for i in range(2, dim_num + 1):
            #
            # 	The bits of the integer POLY(I) gives the form of polynomial I.
            #
            # 	Find the degree of polynomial I from binary encoding.
            #
            j = poly[i - 1]
            m = 0
            while 1:
                j = math.floor(j / 2.)
                if j <= 0:
                    break
                m = m + 1
            #
            # 	Expand this bit pattern to separate components of the logical array INCLUD.
            #
            j = poly[i - 1]
            includ = zeros(m)
            for k in range(m, 0, -1):
                j2 = math.floor(j / 2.)
                includ[k - 1] = j != 2 * j2
                j = j2
            #
            # 	Calculate the remaining elements of row I as explained
            # 	in Bratley and Fox, section 2.
            #
            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l = 1
                for k in range(1, m + 1):
                    l = 2 * l
                    if includ[k - 1]:
                        newv = bitwise_xor(int(newv), int(l * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv
        #
        # 	Multiply columns of V by appropriate power of 2.
        #
        l = 1
        for j in range(maxcol - 1, 0, -1):
            l = 2 * l
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l
        #
        # 	RECIPD is 1/(common denominator of the elements in V).
        #
        recipd = 1.0 / (2 * l)
        lastq = zeros(dim_num)

    seed = int(math.floor(seed))

    if seed < 0:
        seed = 0

    if seed == 0:
        l = 1
        lastq = zeros(dim_num)

    elif seed == seed_save + 1:
        #
        # 	Find the position of the right-hand zero in SEED.
        #
        l = i4_bit_lo0(seed)

    elif seed <= seed_save:

        seed_save = 0
        l = 1
        lastq = zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    elif seed_save + 1 < seed:

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)
    #
    # 	Check that the user is not calling too many times!
    #
    if maxcol < l:
        print('I4_SOBOL - Fatal error!')
        print('	Too many calls!')
        print('	MAXCOL = %d\n' % maxcol)
        print('	L =			%d\n' % l)
        return
    #
    # 	Calculate the new components of QUASI.
    #
    quasi = zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l - 1]))

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]


def i4_uniform(a, b, seed):
    """Returns a scaled pseudorandom I4 scaled to be uniformly distributed
		between a and b.  Seed is a non-zero integer."""
    if seed == 0:
        print('I4_UNIFORM - Fatal error!')
        print('	Input SEED = 0!')

    seed = math.floor(seed)
    a = round(a)
    b = round(b)

    seed = mod(seed, 2147483647)

    if seed < 0:
        seed = seed + 2147483647

    k = math.floor(seed / 127773)

    seed = 16807 * (seed - k * 127773) - k * 2836

    if seed < 0:
        seed = seed + 2147483647

    r = seed * 4.656612875E-10
    #
    # 	Scale R to lie between A-0.5 and B+0.5.
    #
    r = (1.0 - r) * (min(a, b) - 0.5) + r * (max(a, b) + 0.5)
    #
    # 	Use rounding to convert R to an integer between A and B.
    #
    value = round(r)

    value = max(value, min(a, b))
    value = min(value, max(a, b))

    c = value

    return [int(c), int(seed)]


def prime_ge(n):
    """Returns the smallest prime greater than or equal to N."""
    p = max(math.ceil(n), 2)
    while not isprime(p):
        p = p + 1

    return p


def isprime(n):
    """Returns True if N is a prime number, False otherwise."""
    if n != int(n) or n < 1:
        return False
    p = 2
    while p < n:
        if n % p == 0:
            return False
        p += 1
    return True


###------------------------------------------------------------------------------
def sobol(nsamples, dim, skip=1000):
    """
	Generates a Sobol sequence.  Skip a given number of initial points.
	Dim <= 40.
	"""
    r = zeros((dim, nsamples))
    for j in range(1, nsamples + 1):
        seed = skip + j - 2
        [r[0:dim, j - 1], seed] = i4_sobol(dim, seed)
    return r.T
