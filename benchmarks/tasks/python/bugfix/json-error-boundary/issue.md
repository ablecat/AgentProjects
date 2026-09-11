# Record decoding masks unexpected failures

`tasklib.records.decode_record()` should translate malformed JSON syntax into
the documented `ValueError("record is not valid JSON")`.  It currently catches
every `Exception`, so decoder outages and caller type errors are reported as if
the user supplied malformed JSON.  This hides operational failures and makes
diagnosis difficult.

Narrow the exception boundary so only the JSON syntax error is translated.
Keep the existing error and object-validation messages unchanged.
