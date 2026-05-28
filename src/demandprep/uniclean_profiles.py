from __future__ import annotations

from typing import List

from SampleScrubber.cleaner.multiple import AttrRelation
from SampleScrubber.cleaner.single import Date, DisguisedMissHandler, Number, Outlier, Pattern


def build_cleaners(profile: str):
    if profile == "beers":
        return [
            Number("ounces", name="Number_ounces"),
            Number("abv", name="Outlier_abv"),
            AttrRelation(["brewery_id"], ["brewery_name"], "0"),
            AttrRelation(["brewery_id"], ["city"], "1"),
            AttrRelation(["brewery_id"], ["state"], "2"),
            AttrRelation(["beer_name", "brewery_id"], ["abv"], "fd_abv"),
            AttrRelation(["beer_name", "brewery_id"], ["ibu"], "fd_ibu"),
        ]
    if profile == "hospitals":
        return [
            AttrRelation(["Condition", "MeasureName"], ["HospitalType"], "2"),
            AttrRelation(["HospitalName", "PhoneNumber", "HospitalOwner"], ["State"], "3"),
            AttrRelation(["HospitalName"], ["ZipCode"], "4"),
            AttrRelation(["HospitalName"], ["PhoneNumber"], "5"),
            AttrRelation(["MeasureCode"], ["MeasureName"], "6"),
            AttrRelation(["MeasureCode"], ["Stateavg"], "7"),
            AttrRelation(["ProviderNumber"], ["HospitalName"], "8"),
            AttrRelation(["MeasureCode"], ["Condition"], "9"),
            AttrRelation(["HospitalName"], ["Address1"], "10"),
            AttrRelation(["HospitalName"], ["HospitalOwner"], "11"),
            AttrRelation(["City"], ["CountyName"], "12"),
            AttrRelation(["ZipCode"], ["EmergencyService"], "13"),
            AttrRelation(["HospitalName"], ["City"], "14"),
        ]
    if profile == "flights":
        pattern = r"^(0[1-9]|1[0-2]):[0-5][0-9]\s?(AM|PM)$"
        return [
            Pattern("sched_dep_time", pattern, "0"),
            Pattern("act_dep_time", pattern, "1"),
            Pattern("sched_arr_time", pattern, "2"),
            Date("sched_dep_time", "%I:%M %p", "5"),
            Date("act_dep_time", "%I:%M %p", "6"),
            Date("sched_arr_time", "%I:%M %p", "7"),
            Date("act_arr_time", "%I:%M %p", "8"),
            AttrRelation(["flight"], ["sched_dep_time"], "3"),
            AttrRelation(["flight"], ["act_dep_time"], "4"),
            AttrRelation(["flight"], ["sched_arr_time"], "9"),
            AttrRelation(["flight"], ["act_arr_time"], "10"),
        ]
    if profile == "rayyan":
        return [
            Outlier("article_title", [], "3"),
            Outlier("journal_title", [], "4"),
            Outlier("author_list", [], "8"),
            Date(
                "journal_issn",
                "%y-%b",
                "5",
                valid_date_pattern=r"^[A-Za-z]{3}-\d{2}$|^\d{2}-[A-Za-z]{3}$",
            ),
            Date(
                "article_pagination",
                "%b-%y",
                "9",
                valid_date_pattern=r"^(?:\d{1}-\d{2}|\d{2}-\d{1})$",
            ),
            Date("article_jcreated_at", "%-m/%-d/%y", currenFormat="%y/%m/%d", name="10"),
            DisguisedMissHandler("article_jvolumn", "-1", "6"),
            DisguisedMissHandler("article_jissue", "-1", "7"),
            AttrRelation(["journal_abbreviation"], ["journal_title"], "0"),
            AttrRelation(["journal_abbreviation"], ["journal_issn"], "1"),
            AttrRelation(["journal_title"], ["journal_issn"], "2"),
        ]
    if profile == "tax":
        return [
            Outlier("fname", [], "11"),
            Outlier("lname", [], "12"),
            Pattern("gender", "[M|F]", "1"),
            Pattern("areacode", "[0-9]{3}", "2"),
            Pattern("state", "[A-Z]{2}", "3"),
            AttrRelation(["zip"], ["state"], "4"),
            AttrRelation(["areacode"], ["state"], "5"),
            AttrRelation(["zip"], ["city"], "6"),
            AttrRelation(["fname"], ["gender"], "7"),
            AttrRelation(["zip", "haschild"], ["childexemp"], "8"),
            AttrRelation(["zip", "maritalstatus"], ["singleexemp"], "9"),
            AttrRelation(["zip", "maritalstatus"], ["marriedexemp"], "10"),
        ]
    raise KeyError(f"Unknown UniClean cleaner profile: {profile}")
