from unittest import mock

_MOCKED_LOG = mock.Mock()
_MOCKED_LOG._log.return_value = _MOCKED_LOG  # pylint: disable=protected-access
