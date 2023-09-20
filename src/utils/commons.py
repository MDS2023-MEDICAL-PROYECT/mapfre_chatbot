
class Utils:
    @classmethod
    def clean_string(cls, input):
        substring = ""

        if input is not None:
            start_pos = str(input).find(" additional_kwargs")
            if start_pos != -1:
                substring = str(input)[8:start_pos]

        return substring
    @classmethod
    def convert_to_string(cls, messages):
        i = 0
        result = ""
        for message in messages:
            if i % 2 == 0:
                result += "-Human: " + clean_string(message) + "\n"
            else:
                result += "-Doctor: " + clean_string(message) + "\n"
            i = i + 1

        return result