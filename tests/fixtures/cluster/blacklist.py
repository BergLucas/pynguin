#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

# Uses function from blacklisted module
from tempfile import gettempdirb

import tests.fixtures.cluster.blacklist_transitive as bl_tr


def foo():
    gettempdirb()
    bl_tr.bar()
